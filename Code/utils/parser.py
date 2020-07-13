import re
import numpy as np

def pad(text, MAXLEN):
    text = text[:MAXLEN]
    res = np.pad(text, [(0, MAXLEN - text.shape[0]), (0, 0)], mode="constant")
    return res


def parse(parser, df, embedding_dim, MAXLEN):
    text_by_user = {}
    text_by_item = {}
    words_by_item = {}
    notParsedWords = []
    drop_user = []
    drop_item = []

    for userid, itemid, text in zip(df["reviewerID"].tolist(), df["asin"].tolist(), df["reviewText"].tolist()):
        if userid not in text_by_user:
            text_by_user[userid] = []
        if itemid not in text_by_item:
            text_by_item[itemid] = []
        if itemid not in words_by_item:
            words_by_item[itemid] = []

        if text != text:
            continue

        words = text.split()
        parsed_text = []
        parsed_words = []
        for word in words:
            extracteds = re.findall(r'[a-zA-Z\']+', word)
            for extracted in extracteds:
                try:
                    parsed_text.append(parser[extracted][:embedding_dim])
                    parsed_words.append(extracted)
                except:
                    try:
                        parsed_text.append(parser[extracted.lower()][:embedding_dim])
                        parsed_words.append(extracted)
                    except:
                        notParsedWords.append(extracted)

        text_by_user[userid] += parsed_text
        text_by_item[itemid] += parsed_text
        words_by_item[itemid] += parsed_words
    for key, val in text_by_user.items():
        if len(val) == 0:
            drop_user.append(key)
        else:
            text_by_user[key] = pad(np.array(val), MAXLEN)

    for key, val in text_by_item.items():
        if len(val) == 0:
            drop_item.append(key)
        else:
            text_by_item[key] = pad(np.array(val), MAXLEN)
    for key, val in words_by_item.items():
        words_by_item[key] = words_by_item[key][:MAXLEN]

    notParsedWords = list(set(notParsedWords))
    return text_by_user, text_by_item, words_by_item, notParsedWords, drop_user, drop_item




def parse2(parser, df, MAXLEN):
    words_by_item = {}
    notParsedWords = []
    drop_user = []
    drop_item = []
    for userid, itemid, text in zip(df["reviewerID"].tolist(), df["asin"].tolist(), df["reviewText"].tolist()):
        if itemid not in words_by_item:
            words_by_item[itemid] = []

        if text != text:
            continue

        words = text.split()

        parsed_words = []
        for word in words:
            extracteds = re.findall(r'[a-zA-Z\']+', word)
            for extracted in extracteds:
                try:
                    temp = parser[extracted]
                    parsed_words.append(extracted)
                except:
                    try:
                        temp = parser[extracted.lower()]
                        parsed_words.append(extracted)
                    except:
                        notParsedWords.append(extracted)

        words_by_item[itemid] += parsed_words

    for key, val in words_by_item.items():
        words_by_item[key] = words_by_item[key][:MAXLEN]

    notParsedWords = list(set(notParsedWords))
    return words_by_item, notParsedWords, drop_user, drop_item

def parse3(parser, df):
    words_by_item = {}
    words_by_user = {}
    notParsedWords = []
    drop_user = []
    drop_item = []
    for userid, itemid, text in zip(df["reviewerID"].tolist(), df["asin"].tolist(), df["reviewText"].tolist()):
        if itemid not in words_by_item:
            words_by_item[itemid] = []

        if userid not in words_by_user:
            words_by_user[userid] = []

        if text != text:  # テキストなし
            # print(text, " this row doesn't have text")
            continue

        words = text.split()

        parsed_words = []
        for word in words:
            extracteds = re.findall(r'[a-zA-Z\']+', word)
            for extracted in extracteds:
                try:
                    temp = parser[extracted]
                    parsed_words.append(extracted)
                except:
                    try:
                        temp = parser[extracted.lower()]
                        parsed_words.append(extracted.lower())
                    except:
                        notParsedWords.append(extracted)

        words_by_item[itemid] += parsed_words
        words_by_user[userid] += parsed_words

    notParsedWords = list(set(notParsedWords))
    return words_by_item, words_by_user, notParsedWords, drop_user, drop_item
