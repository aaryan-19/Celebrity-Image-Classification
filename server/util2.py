import requests
import lxml
from bs4 import BeautifulSoup


def getNews(name):
    response = []
    name = name.replace(" ", "-", 10)
    # print("https://timesofindia.indiatimes.com/topic/"+name)
    pagesource = requests.get("https://timesofindia.indiatimes.com/topic/" + name)
    soup = BeautifulSoup(pagesource.content, 'lxml')
    # print(soup.prettify())
    data = soup.find_all("div", {'class': 'uwU81'})
    for i in data:
        temp = {}
        link = i.find("a")
        dv=i.find('div', attrs={'class': 'cOu80'})
        img=dv.find("img")["src"]
        con = i.find("span")
        # print(link["href"])
        response.append({"img":img,"link": link["href"], "heading": con.get_text()})
    # print(response)
    return response


if __name__ == "__main__":
    print(getNews("serena williams"))
