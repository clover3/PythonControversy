import os
import json
import pickle

dir_path = "C:\work\Data\guardian data\crawl_comment"

def load_comments(dir_path):
    cnt = 0
    corpus = []
    for name in os.listdir(dir_path):
        path = os.path.join(dir_path, name)
        if os.path.isfile(path) :
            comment_list = []
            f = open(path, encoding='utf-8')
            j = json.load(f)
            discussion = j['discussion']
            current_page = j['currentPage']
            discussion_id = discussion['key']
            web_url = discussion['webUrl']
            comment_count = j['discussion']['commentCount']
            comments =  j['discussion']['comments']
            for comment in comments:
                comment_id = comment['id']
                comment_body = comment['body']
                comment_list.append((comment_id, comment_body))
                if 'responses' in comment:
                    for response in comment['responses']:
                        res_body = response['body']
                        res_id = response['id']
                        comment_target = int(response['responseTo']['commentId'])
                        comment_list.append((res_id, res_body, comment_target))

            if comment_count > 0:
               corpus.append((discussion_id, web_url, comment_list))

            #print("{} Page{} #comment={} ({})".format(discussion_id, current_page, comment_count, len(comment_list)))

        cnt = cnt + 1
        if cnt % 100 == 0 :
            print(cnt)
    return corpus

pickle.dump(load_comments(dir_path), open("comments.pickle","wb"), protocol=2)