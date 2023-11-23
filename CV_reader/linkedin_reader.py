from linkedin_scraper import Person, actions
from selenium import webdriver
driver = webdriver.Chrome()


email = "wej.purvis@gmail.com"
password = "EFhackathonapples"

driver = webdriver.Chrome()
actions.login(driver, email, password) # if email and password isnt given, it'll prompt in terminal
person = Person("https://www.linkedin.com/in/danielspetrov/", driver=driver, scrape=True)
