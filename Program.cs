// ∴ 星算语 v0.2 · QCPU 驱动核编译引擎（C# 原生 · 完全脱离 Python）
// =================================================================
// 城主指令（2026-09-08）：脱离 Python，用普适性量子驱动器改造编译。
// 本引擎 = QCPU·驱动核（普适量子驱动算法，源自 qcpu_v1 数学）：
//   算力 = 量子寄存器 · 计算 = 量子态演化 · 结果 = 测量坍缩
// 星算语三符（▸ 位置 / ↦ 变换 / ⟶ 流）编译为量子驱动指令序列，
// 由驱动核执行：读取 = 演化后测量，运输 = 流径上完成运算。
// 编译方式：源码 → 字节码 → 驱动核执行；产物为独立 exe，宿主与 Python 无关。
// v0.12+（2026-09-09）：Run 拆为分派链 + 27 个指令方法，撞名根治。
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

static class QCPU {
    // ── 常量表：存算一体符号原语（与 tri_drive_v03 同源）──
    public static readonly Dictionary<string, double> CONSTS = new Dictionary<string, double> {
        { "ℊ", Math.Pow(2, 30) }, { "Ϝ", 28 }, { "K", 1024 }, { "M", Math.Pow(2, 20) }, { "G", Math.Pow(2, 30) },
        { "π", Math.PI }, { "e", Math.E }, { "φ", (1 + Math.Sqrt(5)) / 2 }
    };

    // [0] 量子寄存器：N-qubit 态矢量，Σ|cᵢ|² = 1
    public class QReg {
        public Complex[] State;
        public int N;
        public QReg(int n) { N = n; Reset(); }
        public void Reset() {
            State = new Complex[1 << N];
            State[0] = Complex.One;              // |0⟩^⊗N
        }
        // [1] 均匀叠加 H^⊗N
        public void HadamardAll() {
            double a = 1.0 / Math.Sqrt(1 << N);
            for (int i = 0; i < State.Length; i++) State[i] = new Complex(a, 0);
        }
        // [2] 单比特门（H/X/Z/Pθ）
        public void ApplySingle(Complex[,] U, int q) {
            int d = State.Length, step = 1 << q;
            for (int i = 0; i < d; i += step << 1)
                for (int j = i; j < i + step; j++) {
                    Complex c0 = State[j], c1 = State[j + step];
                    State[j] = U[0, 0] * c0 + U[0, 1] * c1;
                    State[j + step] = U[1, 0] * c0 + U[1, 1] * c1;
                }
        }
        // [2] CNOT（控制 c，目标 t）
        public void Cnot(int c, int t) {
            int d = State.Length;
            var ns = new Complex[d];
            for (int i = 0; i < d; i++)
                ns[((i >> c) & 1) == 1 ? (i ^ (1 << t)) : i] = State[i];
            State = ns;
        }
        // [2b] v0.14b TOFFOLI（双控制 c1,c2，目标 t）：量子与门，两控均 1 才翻转
        public void Toffoli(int c1, int c2, int t) {
            int d = State.Length;
            var ns = new Complex[d];
            for (int i = 0; i < d; i++)
                ns[((i >> c1) & 1) == 1 && ((i >> c2) & 1) == 1 ? (i ^ (1 << t)) : i] = State[i];
            State = ns;
        }
        // [2c] v0.14b SWAP（交换 a,b 两比特）
        public void Swap(int a, int b) {
            int d = State.Length;
            var ns = new Complex[d];
            for (int i = 0; i < d; i++) {
                if (((i >> a) & 1) != ((i >> b) & 1)) ns[i ^ (1 << a) ^ (1 << b)] = State[i];
                else ns[i] = State[i];
            }
            State = ns;
        }
        // [3] 贝尔纠缠 (|00⟩+|11⟩)/√2
        public void BellPair() {
            Reset();
            double a = 1.0 / Math.Sqrt(2);
            State[0] = new Complex(a, 0);
            State[(1 << N) - 1] = new Complex(a, 0);
        }
        // [4] 酉演化 e^{-iHt}（特征分解，H 实对称）：V·diag(e^{-iλt})·V†|ψ⟩
        public void Evolve(double[,] H, double t) {
            int d = State.Length;
            var A = new double[d, d];
            for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) A[i, j] = H[i, j];
            var ev = EigenSym(A);
            var V = ev.Item1; var lam = ev.Item2;
            var tmp = new Complex[d];
            for (int i = 0; i < d; i++) { tmp[i] = Complex.Zero; for (int k = 0; k < d; k++) tmp[i] += V[k, i] * State[k]; }
            for (int i = 0; i < d; i++) tmp[i] *= Complex.FromPolarCoordinates(1, -lam[i] * t);
            var ns = new Complex[d];
            for (int i = 0; i < d; i++) { ns[i] = Complex.Zero; for (int k = 0; k < d; k++) ns[i] += V[i, k] * tmp[k]; }
            State = ns;
        }
        // 雅可比特征分解（实对称）
        static (double[,], double[]) EigenSym(double[,] A) {
            int n = A.GetLength(0);
            var a = new double[n, n]; var v = new double[n, n];
            for (int i = 0; i < n; i++) { for (int j = 0; j < n; j++) a[i, j] = A[i, j]; v[i, i] = 1; }
            for (int iter = 0; iter < 64; iter++) {
                int p = 0, q = 1; double mx = 0;
                for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (Math.Abs(a[i, j]) > mx) { mx = Math.Abs(a[i, j]); p = i; q = j; }
                if (mx < 1e-12) break;
                double app = a[p, p], aqq = a[q, q], apq = a[p, q];
                double theta = 0.5 * Math.Atan2(2 * apq, aqq - app);
                double c = Math.Cos(theta), s = Math.Sin(theta);
                for (int k = 0; k < n; k++) {
                    double akp = a[k, p], akq = a[k, q];
                    a[k, p] = a[p, k] = c * akp - s * akq;
                    a[k, q] = a[q, k] = s * akp + c * akq;
                }
                a[p, p] = c * c * app - 2 * s * c * apq + s * s * aqq;
                a[q, q] = s * s * app + 2 * s * c * apq + c * c * aqq;
                a[p, q] = a[q, p] = 0;
                for (int k = 0; k < n; k++) { double vkp = v[k, p], vkq = v[k, q]; v[k, p] = c * vkp - s * vkq; v[k, q] = s * vkp + c * vkq; }
            }
            var lam = new double[n];
            for (int i = 0; i < n; i++) lam[i] = a[i, i];
            return (v, lam);
        }
        // [5] 测量坍缩：P(i)=|⟨i|ψ⟩|²，投影到计算基
        public (int, double[]) Measure(Random rnd) {
            double[] prob = new double[State.Length];
            for (int i = 0; i < State.Length; i++) prob[i] = State[i].Magnitude * State[i].Magnitude;
            double r = rnd.NextDouble(), acc = 0; int o = State.Length - 1;
            for (int i = 0; i < State.Length; i++) { acc += prob[i]; if (r < acc) { o = i; break; } }
            Reset(); State[o] = Complex.One;
            return (o, prob);
        }
        public double Norm() { double s = 0; foreach (var c in State) s += c.Magnitude * c.Magnitude; return Math.Sqrt(s); }
    }

    // [6] 格罗弗搜索原语（通用：N 维搜索空间 + 目标集）
    public static double[] Grover(int N, HashSet<int> marked, int? R0 = null) {
        int M = marked.Count;
        if (M == 0) { var e = new double[N]; for (int i = 0; i < N; i++) e[i] = 1.0 / N; return e; }
        double theta = Math.Asin(Math.Sqrt((double)M / N));
        int R = R0 ?? Math.Max(1, (int)Math.Round(Math.PI / (4 * theta) - 0.5));
        var psi = new Complex[N];
        for (int i = 0; i < N; i++) psi[i] = new Complex(1.0 / Math.Sqrt(N), 0);
        for (int it = 0; it < R; it++) {
            foreach (var m in marked) psi[m] = -psi[m];
            var avg = Complex.Zero; foreach (var c in psi) avg += c; avg /= N;
            for (int i = 0; i < N; i++) psi[i] = 2 * avg - psi[i];
        }
        var prob = new double[N];
        for (int i = 0; i < N; i++) prob[i] = psi[i].Magnitude * psi[i].Magnitude;
        return prob;
    }

    public static Complex[,] H = new Complex[,] { { new Complex(1 / Math.Sqrt(2), 0), new Complex(1 / Math.Sqrt(2), 0) }, { new Complex(1 / Math.Sqrt(2), 0), new Complex(-1 / Math.Sqrt(2), 0) } };
    public static Complex[,] X = new Complex[,] { { Complex.Zero, Complex.One }, { Complex.One, Complex.Zero } };
    public static Complex[,] Z = new Complex[,] { { Complex.One, Complex.Zero }, { Complex.Zero, new Complex(-1, 0) } };
    public static Complex[,] P(double th) => new Complex[,] { { Complex.One, Complex.Zero }, { Complex.Zero, Complex.FromPolarCoordinates(1, th) } };
    // v0.14b 旋转门（角度制：RX(90) = π/2）：单比特任意旋转，与 H/X/Z 合成单比特酉完备集
    public static Complex[,] Rx(double deg) { double h = deg * Math.PI / 360.0, c = Math.Cos(h), s = Math.Sin(h); return new Complex[,] { { c, new Complex(0, -s) }, { new Complex(0, -s), c } }; }
    public static Complex[,] Ry(double deg) { double h = deg * Math.PI / 360.0, c = Math.Cos(h), s = Math.Sin(h); return new Complex[,] { { c, -s }, { s, c } }; }
    public static Complex[,] Rz(double deg) { double h = deg * Math.PI / 360.0; return new Complex[,] { { Complex.FromPolarCoordinates(1, -h), Complex.Zero }, { Complex.Zero, Complex.FromPolarCoordinates(1, h) } }; }
}

// ── 星算语编译执行引擎 ──
static class XingLang {
    class Cell {
        public string Name;
        public object Raw;                      // 原值（写入时）：double / string / double[]（v0.5 向量）
        public Func<double, double> Xform;      // 变换链（读时演化，向量位置=逐元素演化）
        public Func<double, double> Self;       // 内禀拍自演化律（读取即推进一拍）
        public bool IsXform;                    // 纯变换位置（▸f = ↦ expr，函数一等公民）
        public bool Moved;                      // Rust 所有权：⟶! 移动后源失效
        public string LazyFrom;                 // Lisp 惰性：⤳ 读时才求值（源位置名）
        public Func<double, double> LazyXform;  // 惰性流的变换
        public List<string> Qcirc;              // Q# 量子门即语言：⟦H(1);CNOT(1,0)⟧
        public string AddrPath;                 // C 零抽象：绑定文件地址（读=取，写=存，存算一体）
        public int AddrKind;                    // v0.6 存算一体地址：0=全量 1=第N行 2=字节区间[M:N)
        public int AddrArg, AddrArg2;
        public string AddrKey;                  // v0.7 JSON 字段路径（channels.0.affinity）
        public string AddrArgVar1, AddrArgVar2; // v0.8 地址参数可变量（§n / [a:b] 用变量名，读时解析）
        public string Dim;                      // v0.10 量纲（GB/MB/B/%/min…，null=无量纲）
        public List<string> SupPos;             // v0.11 叠加：一逻辑名 = 多物理位置（读=全读合流 写=广播）
        public string Entangled;                // v0.11 纠缠：双向镜像对偶名（写一即写二）
        public int Accesses;                    // v0.13 位置网络：读取计数（死位置检测）
        public string XformSrc;                 // v0.13a 快照：变换链源码（加载时重建 Func）
        public string XformRef;                 // v0.13a 快照：变换来自 IsXform 位置引用时记名
        public string LazyExprSrc;              // v0.13a 快照：惰性流变换源码
        public string SelfInitSrc, SelfExprSrc; // v0.13a 快照：内禀拍初始/律源码
        public string JitterSrc;                // v0.15 域抖动 ∿：电子传导路径的模拟（源位置）
        public double JitterAmp;                // 抖动强度（相对幅度 0~1；0=关掉矫正保留微扰）
        public List<string> FuncArgs;           // v0.16 多参函数：参数名表
        public string FuncBody;                 // v0.16 多参函数：函数体源码（调用时参数替换再求值）
        public object Read() {
            Accesses++;
            if (FuncArgs != null) throw new Exception($"星算语：函数位置需调用 {Name}(…)（函数是延迟体，不能直接读）");   // v0.16
            if (JitterSrc != null) {                                            // v0.15 域抖动：传导路径微扰，每次读取 = 一个可能性切片
                if (!Cells.ContainsKey(JitterSrc)) throw new Exception($"星算语：抖动源位置不存在 {JitterSrc}");
                double bv = Cell.ToD(Cells[JitterSrc].Read());
                return bv * (1 + JitterAmp * (JitterRnd.NextDouble() * 2 - 1));
            }
            if (SupPos != null) return SupPos.Select(n => Cell.ToD(Cells[n].Read())).ToArray();   // v0.11 叠加：全读合流
            if (Qcirc != null) return RunCircuit();                          // 量子线路：执行 → 态矢量
            if (AddrPath != null) return AddrRead();                       // 存算一体：地址位置实时取数
            if (Moved) throw new Exception($"星算语：位置 {Name} 已被移动（⟶! 所有权转移，源失效）");
            if (LazyFrom != null) {                                          // 惰性流：此刻才沿链求值
                object v = Cells[LazyFrom].Read();
                if (v is double[] la) return LazyXform != null ? Map(la, LazyXform) : la;  // 惰性向量：逐元素
                return LazyXform != null ? LazyXform(ToD(v)) : v;
            }
            if (Self != null) { Raw = Self(ToD(Raw)); return Raw; }          // 自演化：读 = 演化一拍 + 测量
            if (Raw is double[] a) return Xform != null ? Map(a, Xform) : a; // APL 向量化变换：逐元素
            return Xform != null ? Xform(ToD(Raw)) : Raw;                    // 读取 = 演化后测量
        }
        // v0.6/0.7/0.8 存算一体：地址位置每次读取都实时从存储取数（按需分页，不锁文件，读多少算多少）
        public object AddrRead() {
            if (AddrKind == 3) return JsonNav();                               // v0.7 JSON 字段寻址
            int a1 = AddrArg, a2 = AddrArg2;                                   // v0.8 地址参数变量化：读时解析
            if (AddrArgVar1 != null) a1 = (int)Cell.ToD(Cells[AddrArgVar1].Read());
            if (AddrArgVar2 != null) a2 = (int)Cell.ToD(Cells[AddrArgVar2].Read());
            string s;
            switch (AddrKind) {
                case 1: s = ReadLine(AddrPath, a1); break;                     // 第 N 行：StreamReader 跳过前 N-1 行
                case 2: s = ReadBytes(AddrPath, a1, a2); break;                // 字节区间：FileStream 只取 M:N 段
                default: s = System.IO.File.ReadAllText(AddrPath, Encoding.UTF8).Trim(); break;
            }
            if (double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double n))
                return Xform != null ? Xform(n) : n;
            return s;                                                        // 文字原样（字符串位置，不套变换）
        }
        // v0.7 JSON 字段寻址：path · key1.key2（数组索引用数字）。实时解析，取数即运算。
        public object JsonNav() {
            JsonDocument doc;
            try { doc = JsonDocument.Parse(System.IO.File.ReadAllText(AddrPath, Encoding.UTF8), new JsonDocumentOptions { AllowTrailingCommas = true, CommentHandling = JsonCommentHandling.Skip }); }
            catch (Exception e) { throw new Exception($"星算语：JSON 解析失败 → {AddrPath}（{e.Message}）"); }
            using (doc) {
                JsonElement cur = doc.RootElement;
                foreach (var p in AddrKey.Split('.')) {
                    if (int.TryParse(p, out int idx)) cur = cur[idx];
                    else cur = cur.GetProperty(p);
                }
                object val;
                switch (cur.ValueKind) {
                    case JsonValueKind.Number: val = cur.GetDouble(); break;
                    case JsonValueKind.String: val = cur.GetString() ?? ""; break;
                    case JsonValueKind.True: val = 1.0; break;
                    case JsonValueKind.False: val = 0.0; break;
                    default: val = cur.ToString(); break;
                }
                if (val is double dn) return Xform != null ? Xform(dn) : dn; // 数值可挂读时变换
                return val;                                                   // 字符串原样
            }
        }
        public static string ReadLine(string path, int n) {
            using (var sr = new StreamReader(path, Encoding.UTF8)) {
                string? line = null;
                for (int i = 1; i <= n; i++) { line = sr.ReadLine(); if (line == null) break; }
                return (line ?? "").Trim();
            }
        }
        public static string ReadBytes(string path, int m, int n2) {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite)) {
                bool bom = fs.ReadByte() == 0xEF && fs.ReadByte() == 0xBB && fs.ReadByte() == 0xBF;
                long off = (bom ? 3 : 0) + m;                                // BOM 是编码标记不是数据：区间跳过
                fs.Seek(off, SeekOrigin.Begin);
                byte[] buf = new byte[Math.Max(0, n2 - m)];
                int got = fs.Read(buf, 0, buf.Length);
                return Encoding.UTF8.GetString(buf, 0, got).Trim('\0');
            }
        }
        // APL 拟合：变换链作用到向量 = 逐元素映射（向量化变换）
        public static double[] Map(double[] a, Func<double, double> f) {
            var r = new double[a.Length];
            for (int i = 0; i < a.Length; i++) r[i] = f(a[i]);
            return r;
        }
        public string RunCircuit() {
            var gates = new List<string>(Qcirc);                              // v0.12 动态 qubit：按线路最大下标定寄存器宽
            for (int i = 0; i < gates.Count; i++) {
                string g = gates[i];
                if (g.StartsWith("@")) { gates.InsertRange(i + 1, Cells[g.Substring(1)].Qcirc); continue; }  // @子线路引用（原地展开，保持门序）
            }
            int nq = 1;
            foreach (var g in gates) {
                if (g.StartsWith("@")) continue;                              // 子线路引用标记：门已原地展开，扫描时跳过
                var p = g.Split('(', ')', ',');
                string op = p[0].Trim();
                if (op == "SWAP") { int a = int.Parse(p[1].Trim()), b = int.Parse(p[2].Trim()); if (a + 1 > nq) nq = a + 1; if (b + 1 > nq) nq = b + 1; }
                else if (op == "TOFFOLI") { int a = int.Parse(p[1].Trim()), b = int.Parse(p[2].Trim()), t = int.Parse(p[3].Trim()); if (a + 1 > nq) nq = a + 1; if (b + 1 > nq) nq = b + 1; if (t + 1 > nq) nq = t + 1; }
                else if (op == "P" || op == "RX" || op == "RY" || op == "RZ") { int qb = int.Parse(p[2].Trim()); if (qb + 1 > nq) nq = qb + 1; }   // 参数门格式 门(θ,q)
                else {
                    int qb = int.Parse(p[1].Trim());
                    if (qb + 1 > nq) nq = qb + 1;
                    if (op == "CNOT") { int t = int.Parse(p[2].Trim()); if (t + 1 > nq) nq = t + 1; }
                }
            }
            var reg = new QCPU.QReg(nq);
            for (int i = 0; i < gates.Count; i++) {
                string g = gates[i];
                if (g.StartsWith("@")) continue;                              // 子线路引用标记：跳过（门已展开）
                var p = g.Split('(', ')', ',');
                string op = p[0].Trim();
                int qb = int.Parse(p[1].Trim());
                if (op == "H") reg.ApplySingle(QCPU.H, qb);
                else if (op == "X") reg.ApplySingle(QCPU.X, qb);
                else if (op == "Z") reg.ApplySingle(QCPU.Z, qb);
                else if (op == "CNOT") reg.Cnot(qb, int.Parse(p[2].Trim()));
                else if (op == "P") { int qr = int.Parse(p[2].Trim()); reg.ApplySingle(QCPU.P(double.Parse(p[1].Trim(), CultureInfo.InvariantCulture)), qr); }
                else if (op == "RX") { int qr = int.Parse(p[2].Trim()); reg.ApplySingle(QCPU.Rx(double.Parse(p[1].Trim(), CultureInfo.InvariantCulture)), qr); }
                else if (op == "RY") { int qr = int.Parse(p[2].Trim()); reg.ApplySingle(QCPU.Ry(double.Parse(p[1].Trim(), CultureInfo.InvariantCulture)), qr); }
                else if (op == "RZ") { int qr = int.Parse(p[2].Trim()); reg.ApplySingle(QCPU.Rz(double.Parse(p[1].Trim(), CultureInfo.InvariantCulture)), qr); }
                else if (op == "SWAP") reg.Swap(int.Parse(p[1].Trim()), int.Parse(p[2].Trim()));
                else if (op == "TOFFOLI") reg.Toffoli(int.Parse(p[1].Trim()), int.Parse(p[2].Trim()), int.Parse(p[3].Trim()));
            }
            var sb = new StringBuilder();
            for (int i = 0; i < reg.State.Length; i++) {
                double a = reg.State[i].Magnitude;                            // v0.14b 显示概率幅模：RZ 相位门实部 0 不丢项
                if (Math.Abs(a) > 1e-6) { if (sb.Length > 0) sb.Append(" + "); sb.Append($"{a:F3}|{Convert.ToString(i, 2).PadLeft(reg.N, '0')}⟩"); }
            }
            return sb.ToString();
        }
        public void WriteBack(object v) { WriteBack(v, false); }
        public void WriteBack(object v, bool fromEnt) {
            if (SupPos != null) {                                             // v0.11 叠加：广播写（一写多落）
                foreach (var n in SupPos) Cells[n].WriteBack(v, fromEnt);
                return;
            }
            Raw = v;
            if (AddrPath == null) { SyncEnt(v, fromEnt); return; }            // 纯内存位置：仅更新内存值（不落盘）
            string s = v is double d ? d.ToString("G17", CultureInfo.InvariantCulture)
                    : v is double[] arr ? string.Join(",", arr)
                    : v.ToString();                                             // v0.5：字符串/向量写盘
            if (AddrKind == 3) {                                              // v0.9 JSON 字段写回：改字段不重写整个结构（JsonNode 原地改）
                JsonNode node = JsonNode.Parse(System.IO.File.ReadAllText(AddrPath, Encoding.UTF8));
                var cur = node;
                var parts = AddrKey.Split('.');
                for (int i = 0; i < parts.Length - 1; i++)
                    cur = int.TryParse(parts[i], out int idx) ? cur[idx] : cur[parts[i]];
                string lk = parts[parts.Length - 1];
                JsonNode val;
                if (v is double dd) val = JsonValue.Create(dd);
                else if (v is double[] ja) val = new JsonArray(ja.Select(x => JsonValue.Create(x)).ToArray());
                else val = JsonValue.Create(v.ToString());                    // 字符串/其他：原样
                if (int.TryParse(lk, out int li)) cur[li] = val; else cur[lk] = val;
                File.WriteAllText(AddrPath, node.ToJsonString(new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping }), new UTF8Encoding(true)); // 中文不转义
            }
            else if (AddrKind == 1) {                                         // 第 N 行写回：改行不整覆
                var lines = File.ReadAllLines(AddrPath, Encoding.UTF8).ToList();
                while (lines.Count < AddrArg) lines.Add("");
                lines[AddrArg - 1] = s;
                File.WriteAllLines(AddrPath, lines, new UTF8Encoding(true));
            }
            else if (AddrKind == 2) {                                         // 字节区间写回：Seek 覆写（跳过 BOM）
                using (var fs = new FileStream(AddrPath, FileMode.Open, FileAccess.Write, FileShare.ReadWrite)) {
                    bool bom = fs.ReadByte() == 0xEF && fs.ReadByte() == 0xBB && fs.ReadByte() == 0xBF;
                    byte[] b = Encoding.UTF8.GetBytes(s);
                    fs.Seek((bom ? 3 : 0) + AddrArg, SeekOrigin.Begin);
                    fs.Write(b, 0, Math.Min(b.Length, AddrArg2 - AddrArg));   // 不越区间边界
                }
            }
            else File.WriteAllText(AddrPath, s, new UTF8Encoding(true));       // UTF-8 BOM：记事本/工具读盘不乱码
            SyncEnt(v, fromEnt);                                               // v0.11 纠缠：镜像写（fromEnt 防递归）
        }
        private void SyncEnt(object v, bool fromEnt) {
            if (!fromEnt && !string.IsNullOrEmpty(Entangled) && Cells.ContainsKey(Entangled))
                Cells[Entangled].WriteBack(v, true);
        }
        public static double ToD(object o) {
            if (o is double d) return d;
            throw new Exception("星算语：此位置不是数值（无法参与运算）");
        }
        public static double[] ToV(object o) {
            if (o is double[] a) return a;
            throw new Exception("星算语：此位置不是向量（∑/μ 需要数组）");
        }
        public static string ToStr(object o) {                                // v0.8 ∂ 数字提取：统一转字符串
            if (o is string s) return s;
            if (o is double d) return d.ToString("G17", CultureInfo.InvariantCulture);
            if (o is double[] a) return string.Join(",", a);
            if (o is string[] sa) return string.Join(",", sa);                 // v0.12 字符串向量
            return o?.ToString() ?? "";
        }
    }
    static Dictionary<string, Cell> Cells = new Dictionary<string, Cell>();
    static Random JitterRnd = new Random(42);                            // v0.15 域抖动：固定种子（传导噪音可复现，回归可控）

    // v0.10 词法抽离：表达式 → token 列表（量纲检查与求值共用）
    // v0.16 完备性：多字符运算符（≥≤≠∧∨）+ 单字符扩展（> < = ¬ , ⟡）+ 函数调用括号
    static string[] OPS2 = { "≥", "≤", "≠", "∧", "∨" };
    static List<string> Tokenize(string s) {
        s = s.Trim();
        var toks = new List<string>();
        for (int i = 0; i < s.Length; i++) {
            char c = s[i];
            if (c == ' ' || c == '\t') continue;
            if (c == '@' || c == '+' || c == '-' || c == '*' || c == '/' || c == '^' || c == '(' || c == ')'
                || c == '>' || c == '<' || c == '=' || c == '¬' || c == ',' || c == '⟡') { toks.Add(c.ToString()); continue; }
            if (char.IsDigit(c) || c == '.') { int j = i; while (j < s.Length && (char.IsDigit(s[j]) || s[j] == '.')) j++; toks.Add(s.Substring(i, j - i)); i = j - 1; continue; }
            if (char.IsLetter(c) || c == '_') { int j = i; while (j < s.Length && (char.IsLetterOrDigit(s[j]) || s[j] == '_')) j++; toks.Add(s.Substring(i, j - i)); i = j - 1; continue; } // v0.7 变量引用
            bool matched = false;
            foreach (var op in OPS2) {                                                                     // v0.16 多字符运算符优先
                if (i + op.Length <= s.Length && s.Substring(i, op.Length) == op) { toks.Add(op); i += op.Length - 1; matched = true; break; }
            }
            if (matched) continue;
            foreach (var kv in QCPU.CONSTS) {
                if (i + kv.Key.Length <= s.Length && s.Substring(i, kv.Key.Length) == kv.Key) { toks.Add(kv.Key); i += kv.Key.Length - 1; matched = true; break; }
            }
            if (!matched) throw new Exception($"星算语：无法识别的符号 '{c}'");
        }
        return toks;
    }
    // v0.10 量纲递归下降检查（镜像 ParseExpr 结构）：加减两侧量纲非空且不同 → 报错；乘除同量纲继承、异量纲导出
    // 返回表达式结果量纲（单变量引用继承源量纲）
    static string DimOf(List<string> toks) {
        int p = 0;
        Func<string> expr = null, term = null, factor = null, atom = null;
        Func<string> cmp = null, logic = null;
        logic = () => { cmp(); while (p < toks.Count && (toks[p] == "∧" || toks[p] == "∨")) { p++; cmp(); } return null; };   // v0.16 布尔无量纲
        cmp = () => { expr(); while (p < toks.Count && (toks[p] == ">" || toks[p] == "<" || toks[p] == "≥" || toks[p] == "≤" || toks[p] == "=" || toks[p] == "≠")) { p++; expr(); } return null; };   // v0.16 比较无量纲
        expr = () => {
            string v = term();
            while (p < toks.Count && (toks[p] == "+" || toks[p] == "-")) {
                string op = toks[p++]; string r = term();
                if (v != null && r != null && v != r) throw new Exception($"星算语：量纲不匹配 {v} {op} {r}");
                v = v ?? r;
            }
            return v;
        };
        term = () => {
            string v = factor();
            while (p < toks.Count && (toks[p] == "*" || toks[p] == "/")) {
                p++; string r = factor();
                v = v != null && r != null && v != r ? null : (v ?? r);       // 异量纲乘除 → 导出量纲（不追踪复合）
            }
            return v;
        };
        factor = () => {
            if (p < toks.Count && toks[p] == "-") { p++; return factor(); }             // v0.16 一元负号：量纲不变
            string v = atom();
            if (p < toks.Count && toks[p] == "^") { p++; factor(); return v; } // 幂保留左量纲（简化）
            return v;
        };
        atom = () => {
            string t = toks[p++];
            if (t == "@") return null;
            if (t == "¬") return atom();                                     // v0.16 非：无量纲
            if (t == "⟡") { cmp(); if (p < toks.Count && toks[p] == ",") p++; cmp(); if (p < toks.Count && toks[p] == ",") p++; cmp(); return null; }   // v0.16 真值选择：无量纲
            if (t == "(") { string v = logic(); if (p < toks.Count && toks[p] == ")") p++; return v; }   // v0.16 括号内完整表达式
            if (Cells.ContainsKey(t)) return Cells[t].Dim;
            return null;
        };
        return logic();
    }
    // 变换表达式递归下降解析：@ = 原值，支持 + - * / ^ ( ) 数字 常量
    // v0.16 完备性：结构验证层 = 逻辑 → 比较 → 算术；¬ / ⟡ / 多参函数调用结构同镜像
    static Func<double, double> ParseExpr(string s) {
        int pos = 0;
        s = s.Trim();
        for (int i = 0; i < s.Length; i++) if (s[i] == '@') { /* 占位符，解析时特殊处理 */ }
        var toks = Tokenize(s);
        int p = 0;
        Func<double> expr = null;
        Func<double> term = null, factor = null, atom = null;
        Func<double> cmp = null, logic = null;
        logic = () => { cmp(); while (p < toks.Count && (toks[p] == "∧" || toks[p] == "∨")) { p++; cmp(); } return 0.0; };
        cmp = () => { expr(); while (p < toks.Count && (toks[p] == ">" || toks[p] == "<" || toks[p] == "≥" || toks[p] == "≤" || toks[p] == "=" || toks[p] == "≠")) { p++; expr(); } return 0.0; };
        expr = () => {
            double v = term();
            while (p < toks.Count && (toks[p] == "+" || toks[p] == "-")) {
                bool add = toks[p++] == "+"; double r = term(); v = add ? v + r : v - r;
            }
            return v;
        };
        term = () => {
            double v = factor();
            while (p < toks.Count && (toks[p] == "*" || toks[p] == "/")) {
                bool mul = toks[p++] == "*"; double r = factor(); v = mul ? v * r : v / r;
            }
            return v;
        };
        factor = () => { double v = atom(); if (p < toks.Count && toks[p] == "^") { p++; v = Math.Pow(v, factor()); } return v; };
        atom = () => {
            if (p >= toks.Count) throw new Exception("星算语：表达式不完整");
            string t = toks[p++];
            if (t == "-") return -factor();                                              // v0.16 一元负号
            if (t == "@") return 0.0; // 占位，稍后替换
            if (t == "¬") { atom(); return 0.0; }                                   // v0.16 非
            if (t == "⟡") { logic(); if (p < toks.Count && toks[p] == ",") p++; else throw new Exception("星算语：⟡ 缺逗号"); logic(); if (p < toks.Count && toks[p] == ",") p++; else throw new Exception("星算语：⟡ 缺逗号"); logic(); return 0.0; }   // v0.16 真值选择
            if (t == "(") { double v = logic(); if (p < toks.Count && toks[p] == ")") p++; else throw new Exception("星算语：缺右括号"); return v; }   // v0.16 括号内完整表达式
            if (QCPU.CONSTS.ContainsKey(t)) return QCPU.CONSTS[t];
            if (Cells.ContainsKey(t)) {
                var cc = Cells[t];
                if (cc.FuncArgs != null && p < toks.Count && toks[p] == "(") {       // v0.16 函数调用结构（内部求值运行期兜底）
                    p++; int d = 0; bool closed = false;
                    while (p < toks.Count) {
                        string tt = toks[p++];
                        if (tt == "(") d++;
                        else if (tt == ")") { if (d == 0) { closed = true; break; } d--; }
                    }
                    if (!closed) throw new Exception("星算语：函数调用缺右括号");
                    return 0.0;
                }
                return 0.0;                            // v0.7 变量引用：结构验证返回 0，运行时实时读
            }
            if (double.TryParse(t, NumberStyles.Float, CultureInfo.InvariantCulture, out double n)) return n;
            throw new Exception($"星算语：未知项 '{t}'");
        };
        // 先编译常数部分（@ 替换为 0 算出结构），再在运行时注入 @
        double baseVal = logic();
        if (p != toks.Count) throw new Exception("星算语：多余符号");
        // 用重新解析 + 占位替换构造运行函数（把 @ 当作 x 的替身）：
        Func<double, double> run = x => {
            // 对表达式以 x 重解析：简单做法——替换 @ 为 x 后逐 token 求值（复用同结构，重跑一遍）
            return EvalTokens(toks, x);
        };
        return run;
    }

    // v0.16 完备性：求值层 = 逻辑（∨ 低于 ∧）→ 比较 → 算术；¬ / ⟡ / 多参函数调用在 atom
    static double EvalTokens(List<string> toks, double x) {
        int p = 0;
        Func<double> expr = null, term = null, factor = null, atom = null;
        Func<double> cmp = null, logic = null;
        logic = () => {
            double v = cmp();
            while (p < toks.Count && (toks[p] == "∧" || toks[p] == "∨")) {
                bool and = toks[p++] == "∧"; double r = cmp();
                bool vb = v != 0, rb = r != 0;
                v = (and ? (vb && rb) : (vb || rb)) ? 1.0 : 0.0;
            }
            return v;
        };
        cmp = () => {
            double v = expr();
            while (p < toks.Count && (toks[p] == ">" || toks[p] == "<" || toks[p] == "≥" || toks[p] == "≤" || toks[p] == "=" || toks[p] == "≠")) {
                string op = toks[p++]; double r = expr();
                bool t = op == ">" ? v > r : op == "<" ? v < r : op == "≥" ? v >= r : op == "≤" ? v <= r : op == "=" ? v == r : v != r;
                v = t ? 1.0 : 0.0;
            }
            return v;
        };
        expr = () => { double v = term(); while (p < toks.Count && (toks[p] == "+" || toks[p] == "-")) { bool a = toks[p++] == "+"; double r = term(); v = a ? v + r : v - r; } return v; };
        term = () => { double v = factor(); while (p < toks.Count && (toks[p] == "*" || toks[p] == "/")) { bool m = toks[p++] == "*"; double r = factor(); v = m ? v * r : v / r; } return v; };
        factor = () => { double v = atom(); if (p < toks.Count && toks[p] == "^") { p++; v = Math.Pow(v, factor()); } return v; };
        atom = () => {
            string t = toks[p++];
            if (t == "-") return -factor();                                              // v0.16 一元负号
            if (t == "@") return x;
            if (t == "¬") return atom() == 0 ? 1.0 : 0.0;                                        // v0.16 非：非零即真
            if (t == "⟡") {                                                                       // v0.16 真值选择：⟡ cond, a, b
                double c = logic(); if (p < toks.Count && toks[p] == ",") p++; else throw new Exception("星算语：⟡ 缺逗号分隔");
                double a = logic(); if (p < toks.Count && toks[p] == ",") p++; else throw new Exception("星算语：⟡ 缺逗号分隔");
                double b = logic();
                return c != 0 ? a : b;
            }
            if (t == "(") { double v = logic(); if (toks[p] == ")") p++; return v; }   // v0.16 括号内完整表达式
            if (QCPU.CONSTS.ContainsKey(t)) return QCPU.CONSTS[t];
            if (Cells.ContainsKey(t)) {
                var cc = Cells[t];
                if (cc.FuncArgs != null && p < toks.Count && toks[p] == "(") return CallFunc(t, toks, ref p, x);   // v0.16 多参函数调用
                return Cell.ToD(cc.Read());
            }
            return double.Parse(t, CultureInfo.InvariantCulture);
        };
        return logic();
    }

    // v0.16 多参函数调用：收集参数子串 → 独立求值 → 参数替换进函数体再求值
    static double CallFunc(string fname, List<string> toks, ref int p, double x) {
        var fc = Cells[fname];
        p++; // 跳过 (
        var args = new List<List<string>>(); var cur = new List<string>(); int depth = 0; bool closed = false;
        while (p < toks.Count) {
            string tt = toks[p++];
            if (tt == "(") depth++;
            else if (tt == ")") { if (depth == 0) { closed = true; break; } depth--; }
            if (tt == "," && depth == 0) { args.Add(cur); cur = new List<string>(); continue; }
            cur.Add(tt);
        }
        if (!closed) throw new Exception($"星算语：函数调用 {fname} 缺右括号");
        if (cur.Count > 0 || args.Count == 0) args.Add(cur);
        if (args.Count != fc.FuncArgs.Count)
            throw new Exception($"星算语：{fname} 参数个数不符（期望 {fc.FuncArgs.Count}，实得 {args.Count}）");
        var vals = new double[args.Count];
        for (int i = 0; i < args.Count; i++) vals[i] = EvalTokens(args[i], x);
        var btoks = Tokenize(fc.FuncBody);
        var nb = new List<string>(btoks.Count);
        foreach (var tk in btoks) {
            int ai = fc.FuncArgs.IndexOf(tk);
            nb.Add(ai >= 0 ? vals[ai].ToString("G17", CultureInfo.InvariantCulture) : tk);
        }
        return EvalTokens(nb, x);
    }

    // ════════════ A3（v0.12+）Run = 纯分派链 + 27 指令方法 ════════════
    // 每行：去注释 → 去空白 → 按正则分派；每个指令方法独立变量名空间，撞名根治。

    // ════════════ v0.13 位置网络编译（PNC）：不走 IR/字节码/VM 范式 ════════════
    // 编译器不产指令流，产"准备好的位置网络"：
    //   编译期登记（定义表+依赖边）→ 编译期抓悬垂/重复 → 常量编译期坍缩 → 读取=测量
    // 新模式：xinglang.exe --chk <file.xxl> = 编译 + 执行 + 死位置报告（只诊断，不产产物）
    static HashSet<string> Defined = new HashSet<string>();
    public static HashSet<int> ReadLines = new HashSet<int>();          // v0.13a 快照模式：只重放 ! 读取行

    static void CollectExprRefs(string expr, List<string> refs) {
        expr = expr.Trim();
        if (expr.Length == 0) return;
        if (expr.StartsWith("\"") && expr.EndsWith("\"") && expr.Length >= 2) return;      // 字符串字面量：无引用
        if (expr.StartsWith("[")) {                                                       // APL 数组字面量：逐项
            foreach (var p in expr.Trim('[', ']').Split(',')) CollectExprRefs(p, refs);
            return;
        }
        if (expr.StartsWith("⟦") || expr.StartsWith("№") || expr.StartsWith("@")) return; // 量子线路/路径/占位：无位置引用
        try {
            var toks = Tokenize(expr);
            foreach (var t in toks)
                if (Regex.IsMatch(t, "^[A-Za-z_]\\w*$") && !QCPU.CONSTS.ContainsKey(t) && t != "@" && !refs.Contains(t))
                    refs.Add(t);
        } catch { /* 词法失败留给运行期报（编译期不吞正事） */ }
    }

    // 编译期登记：扫描全源码建位置网络（定义表 + 依赖边），悬垂引用/写回未定义编译期报错（带行号）
    static void Precompile(string src) {
        Defined.Clear();
        ReadLines.Clear();
        int ln = 0;
        foreach (var raw in src.Split('\n')) {
            ln++;
            string line = raw;
            int h = line.IndexOf('#');
            if (h >= 0) line = line.Substring(0, h);
            line = line.Trim();
            if (line.Length == 0) continue;
            try {
                Match m;
                string defName = null;
                var refs = new List<string>();
                // 定义类指令：定名 + 引用（精确分支优先，宽泛 ▸ 定义兜底在链尾）
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(.+?)\\s*◆([A-Za-z%]+)$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*::\\s*(.+?)(?:\\s*↦\\s*(.+))?$")).Success) { defName = m.Groups[1].Value; if (m.Groups[3].Success) CollectExprRefs(m.Groups[3].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*:\\s*(\\w+)\\s*↦\\s*(.+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); CollectExprRefs(m.Groups[3].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*:=\\s*([^↦]+?)\\s*↦\\s*(.+)$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); CollectExprRefs(m.Groups[3].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*:=\\s*↦\\s*(.+)$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*↦\\(([^)]+)\\)\\s*(.+)$")).Success) { defName = m.Groups[1].Value; var pset = new HashSet<string>(); foreach (var pa in m.Groups[2].Value.Split(',')) pset.Add(pa.Trim()); var brefs = new List<string>(); CollectExprRefs(m.Groups[3].Value, brefs); foreach (var br in brefs) if (!pset.Contains(br)) refs.Add(br); }   // v0.16 多参函数：参数名是局部量，不入悬垂引用
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*↦\\s*(.+)$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*∂(\\d+)\\s*(\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[3].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⌀\\s*(\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*÷\\s*(\".*?\"|\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); if (m.Groups[3].Value[0] != '"') refs.Add(m.Groups[3].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*∪\\s*(\".*?\"|\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); if (m.Groups[3].Value[0] != '"') refs.Add(m.Groups[3].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*⌕\\s*(\".*?\"|\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); if (m.Groups[3].Value[0] != '"') refs.Add(m.Groups[3].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*⌔\\s*(\\S+?)\\s*,\\s*(\\S+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); CollectExprRefs(m.Groups[3].Value, refs); CollectExprRefs(m.Groups[4].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⟜\\s*(\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*⟐\\s*(.+)$")).Success) { defName = m.Groups[1].Value; foreach (var nm in m.Groups[2].Value.Split(',')) refs.Add(nm.Trim()); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*[∑Σ]\\s*(\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*[μµ]\\s*(\\w+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⟦(.+?)⟧$")).Success) { defName = m.Groups[1].Value; foreach (Match am in Regex.Matches(m.Groups[2].Value, "@(\\w+)")) refs.Add(am.Groups[1].Value); }   // v0.15b 量子线路 @ 子线路引用预检
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*\"(.+)\"$")).Success) { defName = m.Groups[1].Value; }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*\\[([^\\]]+)\\]$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*№\\s*(.+)$")).Success) { defName = m.Groups[1].Value; }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*∿\\s*(\\w+)\\s*,\\s*([\\d.]+)$")).Success) { defName = m.Groups[1].Value; refs.Add(m.Groups[2].Value); }   // v0.15 域抖动：源引用
                else if ((m = Regex.Match(line, "^(\\w+)\\s*⟛\\s*(\\w+)$")).Success) { refs.Add(m.Groups[1].Value); refs.Add(m.Groups[2].Value); }
                else if ((m = Regex.Match(line, "^(\\w+)\\s*⤳\\s*(?:\\(↦\\s*(.+?)\\s*\\)\\s*⤳\\s*)?(\\w+)$")).Success) { defName = m.Groups[3].Value; refs.Add(m.Groups[1].Value); if (m.Groups[2].Success) CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^(\\w+)\\s*⟶!\\s*(?:\\(↦\\s*(.+?)\\s*\\)\\s*⟶!\\s*)?(\\w+)$")).Success) { defName = m.Groups[3].Value; refs.Add(m.Groups[1].Value); if (m.Groups[2].Success) CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^(\\w+)\\s*⟶\\s*(?:\\(↦\\s*(.+?)\\s*\\)\\s*⟶\\s*)?(\\w+)$")).Success) { defName = m.Groups[3].Value; refs.Add(m.Groups[1].Value); if (m.Groups[2].Success) CollectExprRefs(m.Groups[2].Value, refs); }
                else if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(.+)$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); } // 宽泛 ▸ 定义兜底（精确分支全前置）
                else if ((m = Regex.Match(line, "^([A-Za-z]\\w*)\\s*=\\s*(.+)$")).Success) { defName = m.Groups[1].Value; CollectExprRefs(m.Groups[2].Value, refs); } // 写回：目标必须已定义
                else if ((m = Regex.Match(line, "^!\\s*(\\w+)$")).Success) { refs.Add(m.Groups[1].Value); ReadLines.Add(ln); }  // 读取：目标必须已定义
                // 悬垂检查：引用的位置必须已定义（或为常量）
                foreach (var r in refs)
                    if (r.Length > 0 && !Defined.Contains(r) && !QCPU.CONSTS.ContainsKey(r))
                        throw new Exception($"星算语：编译期悬垂引用 → {r}（第 {ln} 行）");
                // 登记定义（写回/纠缠不新增；重定义允许=覆盖语义，不报错）
                if (defName != null && !Defined.Contains(defName)) Defined.Add(defName);
            }
            catch (Exception ex) { throw new Exception($"[第 {ln} 行] {ex.Message}\n    ← {line.Trim()}"); }   // v0.14b 编译期报错也回显源码行
        }
    }

    // --chk 模式：编译 + 执行后输出位置网络报告（死位置 = 定义了从未被读、无副作用）
    public static void PrintNetworkReport() {
        var dead = Cells.Where(kv => kv.Value.Accesses == 0 && kv.Value.AddrPath == null && !kv.Value.IsXform
                                     && kv.Value.Qcirc == null && kv.Value.SupPos == null && kv.Value.Entangled == null
                                     && kv.Value.FuncArgs == null)                        // v0.16 函数位置非死位置（等待调用）
                        .Select(kv => kv.Key).OrderBy(k => k).ToList();
        var alive = Cells.Count - dead.Count;
        Console.WriteLine($"∴ 位置网络：{Cells.Count} 位置 · 活跃 {alive} · 死位置 {dead.Count}");
        foreach (var d in dead) Console.WriteLine($"  [死] {d}");
        if (dead.Count == 0) Console.WriteLine("  零死位置 —— 网络干净");
    }

    // ════════════ v0.13a 快照编译：编译产物 = 程序运行后的状态镜像（.xng）════════════
    // 程序即状态：快照捕获位置网络最终状态，加载后只重放 ! 读取行（定义/写回全部从镜像恢复）。
    // 语义前提：幂等程序（demo 全满足）；不满足时用 --snap 重编译刷新快照。
    class XngCell {
        public string n;
        public object raw;
        public string dim;
        public string addr; public int kind; public int arg; public int arg2;
        public string key; public string argV1; public string argV2;
        public string xform; public bool isXformRef; public bool isXform;
        public string lazyFrom; public string lazyExpr;
        public string selfInit; public string selfExpr;
        public List<string> qc; public List<string> sup; public string ent; public bool moved;
        public string jitterSrc; public double jitterAmp;
        public List<string> fargs; public string fbody;     // v0.16 多参函数
    }
    static JsonSerializerOptions _xngOpts = new JsonSerializerOptions {
        WriteIndented = true,
        IncludeFields = true,                       // XngCell 全字段（System.Text.Json 默认只序列化属性）
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    public static void SnapshotNetwork(string xngPath) {                 // 编译产物：定义态登记表写盘
        File.WriteAllText(xngPath, JsonSerializer.Serialize(_snapDefLog, _xngOpts), new UTF8Encoding(true));
    }

    static bool LoadNetwork(string xngPath) {                            // 加载镜像：定义态登记表 → 快照模式执行
        try { _snapDefLog = JsonSerializer.Deserialize<Dictionary<int, XngCell>>(File.ReadAllText(xngPath, Encoding.UTF8), _xngOpts); }
        catch { return false; }
        return _snapDefLog != null && _snapDefLog.Count > 0;
    }

    public static int CellCount() => Cells.Count;                       // v0.13a 快照统计
    public static void SnapRecordOn() { _snapRecord = true; _snapDefLog.Clear(); }   // --snap：开始记录定义态
    static Dictionary<int, XngCell> _snapDefLog = new();                // v0.13a 定义态登记表（行号 → 定义态）
    static bool _snapRecord = false;                                    // --snap 模式：定义行执行后记录定义态

    static void SnapDef(int ln, string name) {                          // 此刻 Cells[name] = 定义态（刚建未被改）
        var c = Cells[name];
        _snapDefLog[ln] = new XngCell {
            n = c.Name, raw = c.Raw, dim = c.Dim,
            addr = c.AddrPath, kind = c.AddrKind, arg = c.AddrArg, arg2 = c.AddrArg2,
            key = c.AddrKey, argV1 = c.AddrArgVar1, argV2 = c.AddrArgVar2,
            xform = c.XformSrc, isXformRef = c.XformRef != null, isXform = c.IsXform,
            lazyFrom = c.LazyFrom, lazyExpr = c.LazyExprSrc,
            selfInit = c.SelfInitSrc, selfExpr = c.SelfExprSrc,
            qc = c.Qcirc, sup = c.SupPos, ent = c.Entangled, moved = c.Moved,
            jitterSrc = c.JitterSrc, jitterAmp = c.JitterAmp,
            fargs = c.FuncArgs, fbody = c.FuncBody,
        };
    }
    static void RestoreDef(int ln) {                                     // 快照模式：定义行从定义态恢复（等价 Try 效果）
        if (!_snapDefLog.TryGetValue(ln, out var c)) return;
        Cells[c.n] = BuildCell(c);
    }
    static Cell BuildCell(XngCell c) {                                   // 定义态 → Cell（Func 从源码重建）
        var cell = new Cell {
            Name = c.n, AddrPath = c.addr, AddrKind = c.kind, AddrArg = c.arg, AddrArg2 = c.arg2,
            AddrKey = c.key, AddrArgVar1 = c.argV1, AddrArgVar2 = c.argV2, Dim = c.dim,
            Moved = c.moved, LazyFrom = c.lazyFrom, Qcirc = c.qc, SupPos = c.sup, Entangled = c.ent,
            JitterSrc = c.jitterSrc, JitterAmp = c.jitterAmp,
            FuncArgs = c.fargs, FuncBody = c.fbody,
        };
        if (c.raw is JsonElement je) {
            switch (je.ValueKind) {
                case JsonValueKind.Number: cell.Raw = je.GetDouble(); break;
                case JsonValueKind.String: cell.Raw = je.GetString(); break;
                case JsonValueKind.Array:
                    var items = je.EnumerateArray().ToList();
                    if (items.All(x => x.ValueKind == JsonValueKind.String)) cell.Raw = items.Select(x => x.GetString()).ToArray();
                    else cell.Raw = items.Select(x => x.GetDouble()).ToArray();
                    break;
            }
        }
        else if (c.raw != null) cell.Raw = c.raw;
        if (c.xform != null) {
            cell.Xform = c.isXformRef ? Cells[c.xform].Xform : ParseExpr(c.xform);
            cell.IsXform = c.isXform;
        }
        if (c.lazyExpr != null) cell.LazyXform = ParseExpr(c.lazyExpr);
        if (c.selfExpr != null) {
            cell.Self = ParseExpr(c.selfExpr);
            if (c.selfInit != null) cell.Raw = ParseExpr(c.selfInit)(0);
        }
        return cell;
    }
    static void Snap(int ln, Match m, int g = 1) { if (_snapRecord) SnapDef(ln, m.Groups[g].Value); }   // 定义分支后记录定义态
    public static void Run(string src, string srcPath = null) {
        Cells.Clear();
        _snapRecord = false;
        bool snapLoaded = false;
        if (srcPath != null) {                                            // v0.13a 自动增量：源未变直接加载定义态登记表
            string xng = srcPath.Replace(".xxl", ".xng");
            if (File.Exists(xng) && File.GetLastWriteTimeUtc(xng) >= File.GetLastWriteTimeUtc(srcPath))
                snapLoaded = LoadNetwork(xng);
        }
        Precompile(src);                                   // v0.13 编译期登记：悬垂/写回目标编译期报错（带行号）
        int ln = 0;
        foreach (var raw in src.Split('\n')) {
            ln++;
            string line = raw;
            int h = line.IndexOf('#');
            if (h >= 0) line = line.Substring(0, h);
            line = line.Trim();
            if (line.Length == 0) continue;
            if (snapLoaded && _snapDefLog.ContainsKey(ln)) { RestoreDef(ln); continue; }  // 快照模式：定义行从定义态恢复（副作用行照跑）
            try {
                Match m;
                // ① 裸赋值写回（地址位置赋值即写盘；须最先，命中后短路）
                if ((m = Regex.Match(line, "^([A-Za-z]\\w*)\\s*=\\s*(.+)$")).Success && Cells.ContainsKey(m.Groups[1].Value)) { TryWriteBack(m); continue; }
                // ①b v0.14 Forth 影子：⇄ 域交换（栈式语言精髓）
                if ((m = Regex.Match(line, "^(\\w+)\\s*⇄\\s*(\\w+)$")).Success) { TrySwap(m); continue; }
                // ①c v0.15 域抖动：∿ 源, 强度 —— 电路噪音制造可能性的软件预演（去掉矫正、放大微扰）
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*∿\\s*(\\w+)\\s*,\\s*([\\d.]+)$")).Success) { TryJitter(m); Snap(ln, m); continue; }
                // ② v0.11 量子控制流三律
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*⟐\\s*(.+)$")).Success) { TrySup(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⟜\\s*(\\w+)$")).Success) { TryMeas(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^(\\w+)\\s*⟛\\s*(\\w+)$")).Success) { TryEnt(m); continue; }
                // ③ v0.12 字符串原语
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⌀\\s*(\\w+)$")).Success) { TryStrLen(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*÷\\s*(\".*?\"|\\w+)$")).Success) { TryStrSplit(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*∪\\s*(\".*?\"|\\w+)$")).Success) { TryStrJoin(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*⌕\\s*(\".*?\"|\\w+)$")).Success) { TryStrFind(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(\\w+)\\s*⌔\\s*(\\S+?)\\s*,\\s*(\\S+)$")).Success) { TryStrSlice(m); Snap(ln, m); continue; }
                // ④ v0.8 ∂ 数字提取
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*∂(\\d+)\\s*(\\w+)$")).Success) { TryDigit(m); Snap(ln, m); continue; }
                // ⑤ v0.6/0.7 存算一体地址与量尺
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*::\\s*(.+?)(?:\\s*↦\\s*(.+))?$")).Success) { TryAddr(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*№\\s*(.+)$")).Success) { TryLineCount(m); Snap(ln, m); continue; }
                // ⑥ 字面量：字符串 / APL 数组
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*\"(.+)\"$")).Success) { TryStrLit(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*\\[([^\\]]+)\\]$")).Success) { TryArr(m); Snap(ln, m); continue; }
                // ⑦ 向量汇总 / 均值
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*[∑Σ]\\s*(\\w+)$")).Success) { TrySum(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*[μµ]\\s*(\\w+)$")).Success) { TryAvg(m); Snap(ln, m); continue; }
                // ⑦b v0.14 SQL/LINQ 影子：⌿ 声明式筛选（WHERE v > n，一符一义）
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⌿\\s*(\\w+)\\s*,\\s*v\\s*(>|<|≥|≤|=)\\s*(-?\\d+(?:\\.\\d+)?)$")).Success) { TryFilter(m); Snap(ln, m); continue; }
                // ⑧ 量子线路 / 惰性流 / 所有权移动
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*⟦(.+?)⟧$")).Success) { TryQcirc(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^(\\w+)\\s*⤳\\s*(?:\\(↦\\s*(.+?)\\s*\\)\\s*⤳\\s*)?(\\w+)$")).Success) { TryLazy(m); Snap(ln, m, 3); continue; }
                if ((m = Regex.Match(line, "^(\\w+)\\s*⟶!\\s*(?:\\(↦\\s*(.+?)\\s*\\)\\s*⟶!\\s*)?(\\w+)$")).Success) { TryMove(m); Snap(ln, m, 3); continue; }
                // ⑨ 内禀拍自演化（无墙钟）
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*:=\\s*([^\\s↦][^↦]*?)\\s*↦\\s*(.+)$")).Success) { TrySelfEvolveInit(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*:=\\s*↦\\s*(.+)$")).Success) { TrySelfEvolve(m); Snap(ln, m); continue; }
                // ⑩ 变换位置（函数一等公民）/ 量纲声明 / 定义
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*↦\\(([^)]+)\\)\\s*(.+)$")).Success) { TryFunc(m); Snap(ln, m); continue; }   // v0.16 多参函数 ↦(x,y) body（须在单参 ↦ 之前）
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*↦\\s*(.+)$")).Success) { TryXformFn(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(.+?)\\s*◆([A-Za-z%]+)$")).Success) { TryDimDef(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*=\\s*(.+)$")).Success) { TryDefine(m); Snap(ln, m); continue; }
                // ⑪ 挂变换 / 流
                if ((m = Regex.Match(line, "^▸(\\w+)\\s*:\\s*(\\w+)\\s*↦\\s*(.+)$")).Success) { TryAttachXform(m); Snap(ln, m); continue; }
                if ((m = Regex.Match(line, "^(\\w+)\\s*⟶\\s*(?:\\(↦\\s*(.+?)\\s*\\)\\s*⟶\\s*)?(\\w+)$")).Success) { TryFlow(m); Snap(ln, m, 3); continue; }
                // ⑫ 读取
                if ((m = Regex.Match(line, "^!\\s*(\\w+)$")).Success) { TryRead(m); continue; }
                throw new Exception($"星算语：无法解析 → {line}");
            }
            catch (Exception ex) { throw new Exception($"[第 {ln} 行] {ex.Message}"); }
        }
    }

    // ── ① 裸赋值写回：地址绑定位置赋值即写盘（存算一体闭环，v0.5 支持字符串）──
    static void TryWriteBack(Match m) {
        string name = m.Groups[1].Value;
        string rhs = m.Groups[2].Value.Trim();
        // v0.15b 写回抖动域 = 沿传导链写源头（存算一体：微扰只在传导路径上，源头是确定的）
        var wcell = Cells[name];
        while (wcell.JitterSrc != null) { name = wcell.JitterSrc; wcell = Cells[name]; }
        object val;
        if (rhs.StartsWith("\"") && rhs.EndsWith("\"") && rhs.Length >= 2) val = rhs.Substring(1, rhs.Length - 2);
        else {
            var toks = Tokenize(rhs);                                  // v0.10 写回量纲检查：防错单位落盘
            string rdim = DimOf(toks), tdim = wcell.Dim;
            if (tdim != null && rdim != null && tdim != rdim)
                throw new Exception($"星算语：量纲不匹配 写 {rdim} → {tdim}（位置 {name}）");
            val = ParseExpr(rhs)(0);
        }
        wcell.WriteBack(val);
    }

    // ── ①b v0.14 Forth 影子：⇄ 域交换（极简符号，栈式语言精髓）──
    static void TrySwap(Match m) {
        string na = m.Groups[1].Value, nb = m.Groups[2].Value;
        if (!Cells.ContainsKey(na) || !Cells.ContainsKey(nb)) throw new Exception($"星算语：交换位置不存在（{na}/{nb}）");
        var ca = Cells[na]; var cb = Cells[nb];
        object ra = ca.Raw, rb = cb.Raw; string da = ca.Dim, db = cb.Dim;
        ca.Raw = rb; cb.Raw = ra; ca.Dim = db; cb.Dim = da;          // 域随值走：值换了，量纲跟着换
    }

    // ── ①c v0.15 域抖动：∿ 源, 强度 —— 传导路径微扰（每次读取 = 一个可能性切片）──
    static void TryJitter(Match m) {
        string name = m.Groups[1].Value, src = m.Groups[2].Value;
        double amp = double.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture);
        if (amp < 0 || amp > 1) throw new Exception($"星算语：抖动强度须在 0~1 之间（当前 {amp}）");
        if (!Cells.ContainsKey(src)) throw new Exception($"星算语：抖动源位置不存在 {src}");   // v0.15b 定义即校验（双保险）
        Cells[name] = new Cell { Name = name, JitterSrc = src, JitterAmp = amp, Dim = Cells[src].Dim };  // v0.15b 量纲随传导继承
    }

    // ── ② v0.11 三律 ──
    static void TrySup(Match m) {                                        // ⟐ 叠加：一逻辑名 = 多物理位置
        var names = m.Groups[2].Value.Split(',').Select(s2 => s2.Trim()).Where(s2 => s2.Length > 0).ToList();
        foreach (var nm in names)
            if (!Cells.ContainsKey(nm)) throw new Exception($"星算语：叠加源位置不存在 {nm}");
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, SupPos = names };
    }
    static void TryMeas(Match m) {                                       // ⟜ 测量：坍缩出被激活的确定值
        if (!Cells.ContainsKey(m.Groups[2].Value)) throw new Exception($"星算语：测量源位置不存在 {m.Groups[2].Value}");
        object msrc = Cells[m.Groups[2].Value].Read();
        double pick;
        if (msrc is double[] pa) { pick = pa.FirstOrDefault(x => x != 0); if (pick == 0 && pa.Length > 0) pick = pa[0]; }
        else pick = Cell.ToD(msrc);
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = pick };
    }
    static void TryEnt(Match m) {                                        // ⟛ 纠缠：双向镜像（写一即写二）
        string na = m.Groups[1].Value, nb = m.Groups[2].Value;
        if (!Cells.ContainsKey(na) || !Cells.ContainsKey(nb)) throw new Exception($"星算语：纠缠位置不存在（{na}/{nb}）");
        Cells[na].Entangled = nb; Cells[nb].Entangled = na;
    }

    // ── ③ v0.12 字符串原语 ──
    static void TryStrLen(Match m) {                                     // ⌀ 长度（字符数）
        if (!Cells.ContainsKey(m.Groups[2].Value)) throw new Exception($"星算语：长度源位置不存在 {m.Groups[2].Value}");
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = (double)Cell.ToStr(Cells[m.Groups[2].Value].Read()).Length };
    }
    static void TryStrSplit(Match m) {                                   // ÷ 分割（→ 字符串向量）
        if (!Cells.ContainsKey(m.Groups[2].Value)) throw new Exception($"星算语：分割源位置不存在 {m.Groups[2].Value}");
        string sep = m.Groups[3].Value.StartsWith("\"") ? m.Groups[3].Value.Trim('"') : Cell.ToStr(Cells[m.Groups[3].Value].Read());
        string[] parts = Cell.ToStr(Cells[m.Groups[2].Value].Read()).Split(new[] { sep }, StringSplitOptions.None);
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = parts };
    }
    static void TryStrJoin(Match m) {                                    // ∪ 拼接还原
        if (!Cells.ContainsKey(m.Groups[2].Value)) throw new Exception($"星算语：拼接源位置不存在 {m.Groups[2].Value}");
        string sep = m.Groups[3].Value.StartsWith("\"") ? m.Groups[3].Value.Trim('"') : Cell.ToStr(Cells[m.Groups[3].Value].Read());
        object sro = Cells[m.Groups[2].Value].Read();
        string joined = sro is string[] sva ? string.Join(sep, sva) : string.Join(sep, (double[])sro);
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = joined };
    }
    static void TryStrFind(Match m) {                                    // ⌕ 查找（0-based，找不到 -1）
        if (!Cells.ContainsKey(m.Groups[2].Value)) throw new Exception($"星算语：查找源位置不存在 {m.Groups[2].Value}");
        string sub = m.Groups[3].Value.StartsWith("\"") ? m.Groups[3].Value.Trim('"') : Cell.ToStr(Cells[m.Groups[3].Value].Read());
        int idx = Cell.ToStr(Cells[m.Groups[2].Value].Read()).IndexOf(sub, StringComparison.Ordinal);
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = (double)idx };
    }
    static void TryStrSlice(Match m) {                                   // ⌔ 子串/子集 [a,b)
        if (!Cells.ContainsKey(m.Groups[2].Value)) throw new Exception($"星算语：子串源位置不存在 {m.Groups[2].Value}");
        int aa = (int)ParseExpr(m.Groups[3].Value)(0), bb = (int)ParseExpr(m.Groups[4].Value)(0);
        object sro2 = Cells[m.Groups[2].Value].Read();
        object sub;
        if (sro2 is string st) sub = bb <= aa ? "" : st.Substring(Math.Max(0, aa), Math.Min(st.Length, bb) - Math.Max(0, aa));
        else if (sro2 is string[] sva2) sub = sva2.Skip(Math.Max(0, aa)).Take(Math.Max(0, bb - aa)).ToArray();
        else { var dv = (double[])sro2; sub = dv.Skip(Math.Max(0, aa)).Take(Math.Max(0, bb - aa)).ToArray(); }
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = sub };
    }

    // ── ④ v0.8 ∂ 数字提取：取文本第 N 个数字（训练日志/台账解析）──
    static void TryDigit(Match m) {
        int n = int.Parse(m.Groups[2].Value);
        if (!Cells.ContainsKey(m.Groups[3].Value)) throw new Exception($"星算语：∂ 源位置不存在 → {m.Groups[3].Value}");
        var mm = Regex.Matches(Cell.ToStr(Cells[m.Groups[3].Value].Read()), "-?\\d+\\.?\\d*");
        if (n < 1 || n > mm.Count) throw new Exception($"星算语：∂{n} 越界（文本共 {mm.Count} 个数字）");
        Cells[m.Groups[1].Value] = new Cell { Name = m.Groups[1].Value, Raw = double.Parse(mm[n - 1].Value, System.Globalization.CultureInfo.InvariantCulture) };
    }

    // ── ⑤ 存算一体地址 / 量尺 ──
    static void TryAddr(Match m) {                                       // :: 全量/行/区间/JSON字段，可挂变换链（读时算）
        string name = m.Groups[1].Value, pathPart = m.Groups[2].Value.Trim().Trim('"');
        string xformExpr = m.Groups[3].Success ? m.Groups[3].Value.Trim() : null;
        int kind = 0, a1 = 0, a2 = 0; string argVar1 = null, argVar2 = null;
        var am = Regex.Match(pathPart, "^(.+)§(\\d+|\\w+)$");
        if (am.Success) {                                                     // v0.8 行参数可变量
            pathPart = am.Groups[1].Value; kind = 1;
            string p1 = am.Groups[2].Value;
            if (int.TryParse(p1, out int n1)) a1 = n1; else argVar1 = p1;
        }
        else {
            am = Regex.Match(pathPart, "^(.+)\\[(\\d+|\\w+):(\\d+|\\w+)\\]$");
            if (am.Success) {                                                 // v0.8 区间参数可变量
                pathPart = am.Groups[1].Value; kind = 2;
                string p1 = am.Groups[2].Value, p2 = am.Groups[3].Value;
                if (int.TryParse(p1, out int n1)) a1 = n1; else argVar1 = p1;
                if (int.TryParse(p2, out int n2)) a2 = n2; else argVar2 = p2;
            }
            else {
                // v0.7 · = JSON 字段进入：path · key1.key2（数组索引用数字 channels.0.affinity）
                am = Regex.Match(pathPart, "^(.+)\\s*·\\s*([\\w.\\-]+)$");
                if (am.Success) { pathPart = am.Groups[1].Value; kind = 3; a1 = 0; a2 = 0; }
            }
        }
        if (!System.IO.File.Exists(pathPart)) throw new Exception($"星算语：地址不存在 → {pathPart}");
        var acell = new Cell { Name = name, AddrPath = pathPart, AddrKind = kind, AddrArg = a1, AddrArg2 = a2, AddrKey = kind == 3 ? am.Groups[2].Value : null, AddrArgVar1 = argVar1, AddrArgVar2 = argVar2 };
        if (xformExpr != null) { acell.Xform = ParseExpr(xformExpr); acell.XformSrc = xformExpr; }
        Cells[name] = acell;
    }
    static void TryLineCount(Match m) {                                  // № 量尺（先量后取）
        string name = m.Groups[1].Value, path = m.Groups[2].Value.Trim().Trim('"');
        if (!System.IO.File.Exists(path)) throw new Exception($"星算语：地址不存在 → {path}");
        int cnt = 0;
        using (var sr = new StreamReader(path, Encoding.UTF8)) while (sr.ReadLine() != null) cnt++;
        Cells[name] = new Cell { Name = name, Raw = (double)cnt };
    }

    // ── ⑥ 字面量 ──
    static void TryStrLit(Match m) {                                     // 字符串位置
        string name = m.Groups[1].Value, str = m.Groups[2].Value;
        Cells[name] = new Cell { Name = name, Raw = str };
    }
    static void TryArr(Match m) {                                        // APL 数组字面量
        string name = m.Groups[1].Value;
        var parts = m.Groups[2].Value.Split(',');
        var arr = new double[parts.Length];
        for (int i = 0; i < parts.Length; i++) arr[i] = ParseExpr(parts[i].Trim())(0);
        Cells[name] = new Cell { Name = name, Raw = arr };
    }

    // ── ⑦ 向量汇总 / 均值 ──
    static void TrySum(Match m) {                                        // ∑ 求和
        string name = m.Groups[1].Value, sname = m.Groups[2].Value;
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        double s = 0; foreach (var x in Cell.ToV(Cells[sname].Read())) s += x;
        Cells[name] = new Cell { Name = name, Raw = s };
    }
    static void TryAvg(Match m) {                                        // μ 均值
        string name = m.Groups[1].Value, sname = m.Groups[2].Value;
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        var a = Cell.ToV(Cells[sname].Read());
        double s = 0; foreach (var x in a) s += x;
        Cells[name] = new Cell { Name = name, Raw = s / a.Length };
    }
    static void TryFilter(Match m) {                                     // ⌿ 声明式筛选（SQL/LINQ 影子）：从域里按条件筛出子集
        string name = m.Groups[1].Value, sname = m.Groups[2].Value;
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        string cmp = m.Groups[3].Value;
        double num = double.Parse(m.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture);
        var a = Cell.ToV(Cells[sname].Read());
        var hit = new List<double>();
        foreach (var x in a) {
            bool ok = cmp == ">" ? x > num : cmp == "<" ? x < num :
                      cmp == "≥" ? x >= num : cmp == "≤" ? x <= num : Math.Abs(x - num) < 1e-12;
            if (ok) hit.Add(x);
        }
        Cells[name] = new Cell { Name = name, Raw = hit.ToArray() };
    }

    // ── ⑧ 量子线路 / 惰性流 / 所有权移动 ──
    static void TryQcirc(Match m) {                                      // ⟦门;门⟧ 量子线路位置
        string name = m.Groups[1].Value;
        var gates = m.Groups[2].Value.Split(';');
        Cells[name] = new Cell { Name = name, Qcirc = gates.Select(g => g.Trim()).Where(g => g.Length > 0).ToList() };
    }
    static void TryLazy(Match m) {                                       // ⤳ 惰性流（读时才求值 thunk）
        string sname = m.Groups[1].Value, expr = m.Groups[2].Value, dname = m.Groups[3].Value;
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        Cells[dname] = new Cell { Name = dname, LazyFrom = sname, LazyXform = string.IsNullOrEmpty(expr) ? null : ParseExpr(expr), LazyExprSrc = expr };
    }
    static void TryMove(Match m) {                                       // ⟶! 所有权：移动，源失效
        string sname = m.Groups[1].Value, expr = m.Groups[2].Value, dname = m.Groups[3].Value;
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        double v = Cell.ToD(Cells[sname].Read());
        if (!string.IsNullOrEmpty(expr)) v = ParseExpr(expr)(v);
        Cells[sname].Moved = true;                                       // 所有权转移：源失效
        Cells[dname] = new Cell { Name = dname, Raw = v };
    }

    // ── ⑨ 内禀拍自演化（无墙钟，读取即推进一拍）──
    static void TrySelfEvolveInit(Match m) {                             // := 初始 ↦ 律
        string name = m.Groups[1].Value, init = m.Groups[2].Value, expr = m.Groups[3].Value;
        Cells[name] = new Cell { Name = name, Raw = ParseExpr(init)(0), Self = ParseExpr(expr), SelfInitSrc = init, SelfExprSrc = expr };
    }
    static void TrySelfEvolve(Match m) {                                 // := ↦ 律（0 起）
        string name = m.Groups[1].Value, expr = m.Groups[2].Value;
        Cells[name] = new Cell { Name = name, Raw = 0.0, Self = ParseExpr(expr), SelfExprSrc = expr };
    }

    // ── ⑩ 变换位置 / 量纲声明 / 定义 ──
    // v0.16 多参函数：↦(x, y) body —— 函数体延迟求值，调用时参数替换
    static void TryFunc(Match m) {
        string name = m.Groups[1].Value;
        var args = new List<string>();
        foreach (var a in m.Groups[2].Value.Split(',')) { var an = a.Trim(); if (an.Length > 0 && !args.Contains(an)) args.Add(an); else if (an.Length == 0) throw new Exception($"星算语：{name} 参数名不能为空"); }
        string body = m.Groups[3].Value.Trim();
        Tokenize(body);                                                     // 结构预检（词法）
        Cells[name] = new Cell { Name = name, FuncArgs = args, FuncBody = body };
    }

    static void TryXformFn(Match m) {                                    // = ↦ expr 变换位置（函数一等公民）
        string name = m.Groups[1].Value, expr = m.Groups[2].Value;
        Cells[name] = new Cell { Name = name, Raw = 0.0, Xform = ParseExpr(expr), IsXform = true, XformSrc = expr };
    }
    static void TryDimDef(Match m) {                                     // = expr ◆DIM 量纲声明（防算错单位）
        string name = m.Groups[1].Value, expr = m.Groups[2].Value, dim = m.Groups[3].Value;
        var toks = Tokenize(expr);
        DimOf(toks);
        Cells[name] = new Cell { Name = name, Raw = ParseExpr(expr)(0), Dim = dim };
    }
    static void TryDefine(Match m) {                                     // = expr 定义位置（写入原值）
        string name = m.Groups[1].Value, expr = m.Groups[2].Value;
        var toks = Tokenize(expr);
        string dim = DimOf(toks);                                       // 检查量纲 + 结果量纲（单变量引用继承）
        Cells[name] = new Cell { Name = name, Raw = ParseExpr(expr)(0), Xform = null, Dim = dim };
    }

    // ── ⑪ 挂变换 / 流 ──
    static void TryAttachXform(Match m) {                                // : src ↦ expr 挂变换（读时演化）
        string name = m.Groups[1].Value, sname = m.Groups[2].Value, expr = m.Groups[3].Value.Trim();
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        Func<double, double> f;
        bool isXformRef = Cells.ContainsKey(expr) && Cells[expr].IsXform;
        if (isXformRef) f = Cells[expr].Xform;                           // 变换位置引用（函数一等公民）
        else f = ParseExpr(expr);
        object srcV = Cells[sname].Read();
        Cells[name] = new Cell { Name = name, Raw = srcV is double[] arr ? (object)arr : (object)Cell.ToD(srcV), Xform = f, XformSrc = expr, XformRef = isXformRef ? expr : null };
    }
    static void TryFlow(Match m) {                                       // ⟶ 运输时算
        string sname = m.Groups[1].Value, expr = m.Groups[2].Value, dname = m.Groups[3].Value;
        if (!Cells.ContainsKey(sname)) throw new Exception($"星算语：未知位置 {sname}");
        double v = Cell.ToD(Cells[sname].Read());
        if (!string.IsNullOrEmpty(expr)) v = ParseExpr(expr)(v);
        Cells[dname] = new Cell { Name = dname, Raw = v, Xform = null };
    }

    // ── ⑫ 读取：演化后测量坍缩 ──
    static void TryRead(Match m) {
        string name = m.Groups[1].Value;
        if (!Cells.ContainsKey(name)) throw new Exception($"星算语：未知位置 {name}");
        object v = Cells[name].Read();
        string outS;
        if (v is double d) outS = d.ToString("G6", CultureInfo.InvariantCulture);
        else if (v is double[] arr) outS = "[" + string.Join(" ", arr.Select(x => x.ToString("G6", CultureInfo.InvariantCulture))) + "]";
        else if (v is string[] sarr) outS = "[" + string.Join(" ", sarr) + "]";   // v0.12 字符串向量
        else outS = v.ToString();
        Console.WriteLine($"{name} = {outS}");
    }

    // v0.14b 语言元数据：版本 + 原语符号 + 万有影子索引（语言身份证明）
    public static void PrintVersion() {
        Console.WriteLine("∴ 星算语 xinglang · 存算一体语言 v0.16");
        Console.WriteLine("  公理：数据在读取和运输的时候就已经被运算好了");
        Console.WriteLine("  原语：▸ 定义 · :: 地址 · § 行寻址 · № 量尺 · ∂ 数字 · · JSON · ◆ 量纲");
        Console.WriteLine("        ⟐⟛⟜ 可能性三律 · ⟦⟧ 量子线路 · ↦ 变换/函数 · ⤳ 惰性 · ⟶! 所有权 · ⟶ 流");
        Console.WriteLine("        ∑μ⌿ 符号数组 · ⌀÷∪⌕⌔ 字符串 · ⇄ 域交换 · := 自演化 · ∿ 域抖动 · ! 读取");
        Console.WriteLine("  完备性：> < ≥ ≤ = ≠ 比较 · ∧ ∨ ¬ 布尔 · ⟡ 真值选择 · ↦(x,y) 多参函数");
        Console.WriteLine("  影子索引：C/汇编 位置 · Rust 所有权 · Haskell 量纲 · Lisp 同像 · Erlang 流");
        Console.WriteLine("        SQL 筛选 ⌿ · Forth 栈 ⇄ · APL 符号 ∑μ⌿ · Q# 线路 ⟦⟧ · 玻色-费米三律");
        Console.WriteLine("  量子门：H X Z P · RX RY RZ SWAP TOFFOLI · CNOT · 贝尔 · 格罗弗 · 酉演化");
        Console.WriteLine("  编译产物：位置网络（非字节码）· 快照 .xng（定义态登记表）");
    }

    // 量子驱动核自检：H 叠加 → 贝尔纠缠 → 演化 → 测量坍缩 → 格罗弗
    public static void QcSelfTest() {        Console.WriteLine("∴ QCPU·驱动核自检（普适量子驱动算法）");
        var rnd = new Random(42);
        var reg = new QCPU.QReg(2);
        reg.Reset();
        reg.ApplySingle(QCPU.H, 1);     // H 作用高位 qubit1
        reg.Cnot(1, 0);                 // CNOT 控制 1 → 目标 0：|00⟩+|11⟩
        Console.WriteLine($"  贝尔纠缠  |ψ⟩ = ({reg.State[0].Real:F3}|00⟩ + {reg.State[1].Real:F3}|01⟩ + {reg.State[2].Real:F3}|10⟩ + {reg.State[3].Real:F3}|11⟩)");
        double[,] Ham = {
            { 0.50, 0.10, 0.00, 0.00 },
            { 0.10, 0.40, 0.05, 0.00 },
            { 0.00, 0.05, 0.30, 0.10 },
            { 0.00, 0.00, 0.10, 0.20 } };
        reg.Evolve(Ham, 1.0);
        Console.WriteLine($"  酉演化 e^-iHt 后范数 ‖ψ‖ = {reg.Norm():F6}（应为 1）");
        var (o, prob) = reg.Measure(rnd);
        Console.WriteLine($"  测量坍缩 结果 |{o}⟩ 概率 {prob[o]:F3}");
        var g = QCPU.Grover(16, new HashSet<int> { 7 });
        Console.WriteLine($"  格罗弗搜索 目标 |7⟩ 概率 {g[7]:F4}（理论 ≥ {1 - 1.0 / 16:F4}）");
    }
}

class Program {
    static void Main(string[] args) {
        Console.OutputEncoding = Encoding.UTF8;
        if (args.Length == 0) { Console.WriteLine("  用法: xinglang.exe <file.xxl> | --qc | --repl | --chk <file.xxl> | --snap <file.xxl> | --version"); return; }
        if (args[0] == "--version") { XingLang.PrintVersion(); return; }              // v0.14b 语言元数据：版本 + 原语符号 + 影子索引
        if (args[0] == "--qc") { XingLang.QcSelfTest(); return; }
        if (args[0] == "--repl") { Repl(); return; }
        if (args[0] == "--snap") {                                                  // v0.13a 快照编译：全执行 + 定义态登记表写盘（.xng）
            string sp = args[1];
            string src1 = File.ReadAllText(sp, Encoding.UTF8);
            try {
                XingLang.SnapRecordOn();
                XingLang.Run(src1, null);
                XingLang.SnapshotNetwork(sp.Replace(".xxl", ".xng"));
                Console.WriteLine($"∴ 快照已写 → {sp.Replace(".xxl", ".xng")}（{XingLang.CellCount()} 位置定义态）");
            }
            catch (Exception e) { Console.WriteLine($"∴ [异常] {e.Message}"); Environment.Exit(1); }
            return;
        }
        if (args[0] == "--chk") {                                                    // v0.13 位置网络编译检查：悬垂预检 + 执行 + 死位置报告
            string src2 = File.ReadAllText(args[1], Encoding.UTF8);
            try {
                XingLang.Run(src2, null);                                            // 诊断模式全量执行（不加载快照）
                XingLang.PrintNetworkReport();
            }
            catch (Exception e) { Console.WriteLine($"∴ [异常] {e.Message}"); Environment.Exit(1); }
            return;
        }
        string src = File.ReadAllText(args[0], Encoding.UTF8);
        try { XingLang.Run(src, args[0]); }                                          // 自动增量：.xng 新鲜则加载镜像
        catch (Exception e) { Console.WriteLine($"∴ [异常] {e.Message}"); Environment.Exit(1); }
    }
    // REPL：一行一行地编，边立律边测量（天人合一的手谈）
    static void Repl() {
        Console.WriteLine("  ∴ 手谈模式：输一行算一行（exit 退出）");
        var buf = new StringBuilder();
        while (true) {
            Console.Write("  ∴ ");
            string? line = Console.ReadLine();
            if (line == null) break;
            if (line.Trim() == "exit") break;
            if (line.Trim().Length == 0) continue;
            buf.AppendLine(line);
            try { XingLang.Run(buf.ToString()); }
            catch (Exception e) { Console.WriteLine($"  [异常] {e.Message}"); }
        }
    }
}
