from pypinyin import Style


def convert_to_zhuyin(msg, use_ext_dict=True, style=Style.BOPOMOFO):
    # import library
    from pypinyin import pinyin, Style
    import json
    from pypinyin_ext import dict_path

    with open(dict_path) as f:
        myDict = json.loads(f.read())
        # word in ext_dict
        keys = list(myDict.keys())

        converted = [False] * len(msg)
        reservedIndexes = []
        # Initialize with True because the first
        # character of msg may be non-chinese
        prevIsChinese = True
        # check for non-chinese character
        for i in range(len(msg)):
            if not (0x4e00 <= ord(msg[i]) < 0x9fa6):
                converted[i] = True
                # reserve the index of non-chinese character behind
                # the chinese character, since we should copy the
                # value of original_convert for consecutive non-ch-
                # inese character, if run with main you will realize
                if prevIsChinese:
                    reservedIndexes.append(i)
                prevIsChinese = False
            else:
                reservedIndexes.append(i)
                prevIsChinese = True

        # original_convert == reservedIndexes in length
        original_convert = pinyin(msg, style=style)
        if not use_ext_dict:
            return original_convert
        # initialize my_convert list as long as msg
        my_convert = [[] for _ in msg]
        for i in range(len(original_convert)):
            my_convert[reservedIndexes[i]] = original_convert[i]

        # core of extended conversion
        list_of_keys_length = [len(k) for k in keys]
        chk_rounds = max(list_of_keys_length)
        # check from the max length of the word
        # in ext-dict.json
        for i in range(chk_rounds, 0, -1):
            for j in range(len(msg) - i + 1):
                flag = converted[j]
                if not flag:
                    flag2 = not flag
                    for offset in range(1, i):
                        flag2 = flag2 and not converted[j + offset]
                        if not flag2:
                            break

                    if flag2:  # there are consecutive i chinese word maybe covert
                        word = msg[j:j + i]
                        if word in keys:
                            conversion = myDict[word]
                            for k in range(j, j + i):
                                converted[k] = True
                                my_convert[k] = conversion[k - j]

        # filter empty list
        my_convert = [res for res in my_convert if len(res) > 0]

        return my_convert

if __name__ == '__main__':
    msg = '我,要求，睡覺gg。.'
    #print(pinyin(msg, style=Style.BOPOMOFO))
    print(convert_to_zhuyin(msg))