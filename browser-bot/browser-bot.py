from playwright.sync_api import sync_playwright

def login_bot():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=300)
        page = browser.new_page()

        page.goto("https://people.zoho.com/883361910/zp#home/myspace/overview-actionlist")


        page.click('button[type="submit"]')

        page.wait_for_load_state("networkidle")
        print("Logged in")
        print("Title:", page.title())

        browser.close()

login_bot()