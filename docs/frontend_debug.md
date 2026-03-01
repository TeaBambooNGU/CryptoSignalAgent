安装并接入 Playwright：/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend/package.json:24
新增 UI 联调脚本（抓页面截图 + Console + request failed + 4xx/5xx）：/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend/scripts/ui-debug.mjs:1
新增 npm 命令：
ui:debug（无头）：/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend/package.json:10
ui:debug:live（有头，真实打开 Chromium）：/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend/package.json:11
文档已补：/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/README.md:139
调试产物目录已忽略：/Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/.gitignore:26
你现在这样跑（本机可视化）：

终端 A（后端）
cd /Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent && uv run python main.py
终端 B（前端）
cd /Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend && npm run dev
终端 C（联调）
无头抓证据：cd /Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend && npm run ui:debug
可视化调试：cd /Users/teabamboo/Documents/AIplusLLM/CryptoSignalAgent/frontend && npm run ui:debug:live
产出：

截图在 frontend/artifacts/
终端会打印 [browser-console:*]、[browser-pageerror]、[browser-requestfailed]、[browser-http-*]
说明：

我在当前受限沙箱里无法真正拉起浏览器（权限限制），但你本机终端运行 ui:debug:live 可以直接看到真实页面。
你跑完把终端日志 + 截图发我，我就按这套链路直接定位并修 UI bug。