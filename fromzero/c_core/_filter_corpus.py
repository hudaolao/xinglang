# -*- coding: utf-8 -*-
# 过滤语料为干净中文正文
import io, re

src = io.open(r'E:\StarSnow_Home\languages\fromzero\c_core\corpus_xingxue.txt', encoding='utf-8').read()

def keep(ch):
    return ('\u4e00' <= ch <= '\u9fff') or ch in '，。！？；：、""''（）《》—…·\n\t '

clean = ''.join(ch for ch in src if keep(ch))
clean = re.sub(r'\n\s*\n+', '\n', clean).strip()
uniq = set(clean)
print('过滤后长度:', len(clean), '字符')
print('过滤后唯一字符:', len(uniq))
print('=== 过滤后开头 ===')
print(clean[:300])
io.open(r'E:\StarSnow_Home\languages\fromzero\c_core\corpus_xingxue_clean.txt', 'w', encoding='utf-8').write(clean)
