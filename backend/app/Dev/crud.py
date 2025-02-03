import asyncio
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
import os
from dotenv import load_dotenv

load_dotenv()


class HINETLogin:
    email = os.getenv("EMAIL", None)
    password = os.getenv("PASSWORD", None)
    hinet_url = os.getenv("HINET_URL", None)
    driver = None

    @classmethod
    async def start_driver(cls) -> None:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        cls.driver = webdriver.Chrome(options=chrome_options)

    @classmethod
    async def login(cls) -> None:
        await cls.start_driver()
        print("driver started")
        cls.driver.get(cls.hinet_url)

        await asyncio.sleep(1)
        account_input = cls.driver.find_element(By.ID, "i0116")
        account_input.send_keys(cls.email)
        print("email entered")
        next_button = cls.driver.find_element(By.ID, "idSIButton9")
        next_button.click()
        print("next button clicked")
        await asyncio.sleep(2)

        password_input = cls.driver.find_element(By.ID, "i0118")
        password_input.send_keys(cls.password)
        print("password entered")
        signin_button = cls.driver.find_element(By.ID, "idSIButton9")
        signin_button.click()
        print("signin button clicked")
        await asyncio.sleep(2)

        stay_signed_in_button = cls.driver.find_element(By.ID, "idSIButton9")
        stay_signed_in_button.click()
        print("stay signed in button clicked")
        await asyncio.sleep(1)

        try:
            registration_button = cls.driver.find_element(
                By.XPATH, '//button[contains(text(), "Registration")]'
            )
            registration_button.click()
            print("Registration button found and clicked.")
        except NoSuchElementException:
            pass

        await asyncio.sleep(2)
        cls.driver.quit()

    @classmethod
    async def close_driver(cls):
        if cls.driver:
            cls.driver.quit()

    @classmethod
    async def main(cls):
        if cls.email and cls.password and cls.hinet_url:
            await cls.login()
        else:
            raise Exception(
                "Please provide email, password and hinet url in .env file."
            )

    @classmethod
    async def git_pull(cls):
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
