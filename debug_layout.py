import sys
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from src.layout_analyzer import LayoutAnalyzer

config = {
    "layout": {
        "use_gpu": False,
        "show_log": True
    }
}

analyzer = LayoutAnalyzer(config)
res = analyzer.analyze(r"e:\Thuc tap\data\documents\DOC_001\page_001.png")
for b in res["blocks"]:
    print(b)
