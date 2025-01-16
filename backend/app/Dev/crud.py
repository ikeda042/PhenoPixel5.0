import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
import os
from dotenv import load_dotenv

load_dotenv()


class HINETLogin:
    def __init__(self) -> None:
        self.driver = None

        self.email = os.getenv("EMAIL", None)
        self.password = os.getenv("PASSWORD", None)
        self.hinet_url = os.getenv("HINET_URL", None)

    async def start_driver(self) -> None:
        self.chrome_options = Options()
        self.chrome_options.add_argument("--headless")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")

        self.driver = webdriver.Chrome(options=self.chrome_options)

    async def login(self) -> None:
        await self.start_driver()
        print("driver started")
        self.driver.get(self.hinet_url)

        await asyncio.sleep(1)
        account_input = self.driver.find_element(By.ID, "i0116")
        account_input.send_keys(self.email)
        print("email entered")
        next_button = self.driver.find_element(By.ID, "idSIButton9")
        next_button.click()
        print("next button clicked")
        await asyncio.sleep(2)

        password_input = self.driver.find_element(By.ID, "i0118")
        password_input.send_keys(self.password)
        print("password entered")
        signin_button = self.driver.find_element(By.ID, "idSIButton9")
        signin_button.click()
        print("signin button clicked")
        await asyncio.sleep(2)

        stay_signed_in_button = self.driver.find_element(By.ID, "idSIButton9")
        stay_signed_in_button.click()
        print("stay signed in button clicked")
        await asyncio.sleep(1)

        try:
            registration_button = self.driver.find_element(
                By.XPATH, '//button[contains(text(), "Registration")]'
            )

            registration_button.click()

            print("Registration button found and clicked.")
        except NoSuchElementException:
            pass

        await asyncio.sleep(2)
        self.driver.quit()

    async def close_driver(self):
        if self.driver:
            self.driver.quit()

    async def main():
        hinet_login = HINETLogin()
        if hinet_login.email and hinet_login.password and hinet_login.hinet_url:
            await hinet_login.login()
        else:
            raise Exception(
                "Please provide email, password and hinet url in .env file."
            )


async def git_pull():
    process_checkout = await asyncio.create_subprocess_shell(
        "git checkout main",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process_checkout.communicate()

    process_pull = await asyncio.create_subprocess_shell(
        "git pull",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await process_pull.communicate()
