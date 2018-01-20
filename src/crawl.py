import os
import requests
import json
import time

def request_data(article_id):
    #article_id = "world/2016/dec/31/drc-close-to-deal-for-president-joseph-kabila-to-step-down-after-2017-elections"
    url = "https://content.guardianapis.com/"

    apikey = "c13d9515-b19e-412b-b505-994677cc2cf3"


    headers = {
                "api-key": apikey,
                "format": "json",
                "show-blocks":"body",

            }
    res = requests.get(url+article_id, headers)
    if res.status_code != 200:
        print(res.content)
        raise Exception
    return res.content
    #j = json.loads(res.content)


def ask_list(page):
    page_str = str(page)
    url = "http://content.guardianapis.com/search?show-fields=shortUrl&from-date=2016-01-01&to-date=2016-12-31&page-size=200&page="+page_str
    apikey = "c13d9515-b19e-412b-b505-994677cc2cf3"

    headers = {
        "api-key": apikey,
        "format": "json",
    }
    res = requests.get(url, headers)
    j = json.loads(res.content)
    r = j['response']['results']
    id_list = []
    for item in r:
        id = item['id']
        shortUrl = item['fields']['shortUrl']
        id_list.append((id,shortUrl))
    return id_list


def ask_list_with_body(page):
    page_str = str(page)
    url = "http://content.guardianapis.com/search?show-fields=all&from-date=2016-01-01&to-date=2016-12-31&page-size=200&page="+page_str
    apikey = "c13d9515-b19e-412b-b505-994677cc2cf3"

    headers = {
        "api-key": apikey,
        "format": "json",
    }
    res = requests.get(url, headers)
    if res.status_code == 200:
        return res.content
    else :
        return None


def get_ids():
    f = open("guardian_article_ids.txt", "w")
    for page in range(1,552):
        print(page)
        id_list = ask_list(page)
        for id, shortUrl in id_list:
            f.write(id +"\t" + shortUrl + "\n")
        time.sleep(0.1)

    f.close()

def get_article():
    f = open("guardian_article_id.txt", "r")
    article_save_dir = "C:\\work\\Data\\guardian data\\crawl\\"


    cnt =0
    for line in f:
        id = line.strip()
        file_name = id.replace("/","_") + ".json"
        path = os.path.join(article_save_dir, file_name)
        if os.path.exists(path):
            continue
        jdata = request_data(id)
        f = open(path, "wb")
        f.write(jdata)
        f.close()
        time.sleep(0.1)
        cnt = cnt+ 1
        if cnt % 10000 == 0:
            print(cnt)


def get_comment(short_id):
    url_prefix = "http://discussion.guardianapis.com/discussion-api/discussion/"

    url = url_prefix + short_id
    res = requests.get(url)
    if res.status_code == 200 :
        return res.content
    else:
        return None

def crawl_comments():
    save_dir = "C:\\work\\Data\\guardian data\\crawl_comment\\"

    f_log = open("comment_log.txt", "r")
    crawled_id = set()
    if f_log :
        crawled_id = set([line.split("\t")[0] for line in f_log])
        print("{} comments already crawled".format(len(crawled_id)))
        f_log.close()

    f_log = open("comment_log.txt", "a")
    cnt = 0
    short_id_list = [line.split("\t")[1][14:].strip() for line in open("guardian_article_ids.txt", "r")]
    for short_id in short_id_list:
        assert(short_id[0] == '/')
        if short_id in crawled_id:
            continue
        file_name = short_id.replace("/","_") + ".json"
        path = os.path.join(save_dir, file_name)
        data = get_comment(short_id)
        if data:
            f = open(path, "wb")
            f.write(data)
            f.close()
            time.sleep(0.1)
            cnt = cnt + 1
            if cnt % 100 == 0:
                print(cnt)
            f_log.write(short_id + "\tsuccess\n" )
        else:
            f_log.write(short_id + "\tnot exists\n")

def crawl_by_list():
    for page in range(1, 552):
        print(page)
        content = ask_list_with_body(page)
        save_dir = "C:\\work\\Data\\guardian data\\list_crawl\\"
        path = save_dir + str(page) + ".json"
        open(path,"wb").write(content)

        time.sleep(0.1)


#get_ids()
#get_article("/p/5tt8j")
#crawl_comments()
crawl_by_list()
