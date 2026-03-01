import { mkdir } from "node:fs/promises";
import path from "node:path";
import { chromium } from "playwright";

function hasFlag(flag) {
  return process.argv.includes(flag);
}

function readArg(name, fallback) {
  const index = process.argv.findIndex((arg) => arg === name);
  if (index >= 0 && process.argv[index + 1]) {
    return process.argv[index + 1];
  }
  return fallback;
}

const url = readArg("--url", "http://127.0.0.1:5173");
const runAgent = hasFlag("--run-agent");
const headless = hasFlag("--headless");
const waitMs = Number(readArg("--wait-ms", "8000"));

const artifactDir = path.resolve(process.cwd(), "artifacts");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const firstShot = path.join(artifactDir, `ui-debug-${stamp}-initial.png`);
const secondShot = path.join(artifactDir, `ui-debug-${stamp}-after-action.png`);

await mkdir(artifactDir, { recursive: true });

let browser;
try {
  browser = await chromium.launch({ headless });
  const context = await browser.newContext({
    viewport: { width: 1536, height: 960 },
  });
  const page = await context.newPage();

  page.on("console", (msg) => {
    console.log(`[browser-console:${msg.type()}] ${msg.text()}`);
  });

  page.on("pageerror", (error) => {
    console.error(`[browser-pageerror] ${error.message}`);
  });

  page.on("requestfailed", (request) => {
    console.error(
      `[browser-requestfailed] ${request.method()} ${request.url()} -> ${request.failure()?.errorText ?? "unknown"}`,
    );
  });

  page.on("response", (response) => {
    if (response.status() >= 400) {
      console.error(`[browser-http-${response.status()}] ${response.request().method()} ${response.url()}`);
    }
  });

  console.log(`[ui-debug] open ${url}`);
  await page.goto(url, { waitUntil: "domcontentloaded", timeout: 30000 });
  await page.waitForTimeout(1200);
  await page.screenshot({ path: firstShot, fullPage: true });
  console.log(`[ui-debug] screenshot: ${firstShot}`);

  if (runAgent) {
    const userIdInput = page.locator('input[placeholder="u001"]').first();
    const queryInput = page.locator('textarea[placeholder="请描述你想要的研报任务"]').first();
    const runButton = page.getByRole("button", { name: /run agent|generating/i });

    if (await userIdInput.count()) {
      await userIdInput.fill("u001");
    }
    if (await queryInput.count()) {
      await queryInput.fill("请给我 BTC 和 ETH 的 24 小时风险信号研报");
    }
    if (await runButton.count()) {
      await runButton.click();
      console.log("[ui-debug] clicked Run Agent");
    }

    await page.waitForTimeout(waitMs);
    await page.screenshot({ path: secondShot, fullPage: true });
    console.log(`[ui-debug] screenshot: ${secondShot}`);
  }

  console.log("[ui-debug] done");
} catch (error) {
  console.error("[ui-debug] launch failed. Please run on local terminal instead of restricted sandbox.");
  if (error instanceof Error) {
    console.error(`[ui-debug] ${error.message}`);
  }
  process.exitCode = 1;
} finally {
  if (browser) {
    await browser.close();
  }
}
