def parse(d, record=None, current='', count=0):
    """
    Parse a dict(may have sub-dict) into another dict which has no sub-dict.
    The original dict is replaced by {key_path: [type, value, n-th sub-dict]
    """
    if not isinstance(record, dict):
        record = dict()
    for k in d.keys():
        # private attribute
        if isinstance(k, str) and k.startswith('__'):
            continue
        v = d[k]
        t = type(v)
        p = current + k
        if isinstance(v, dict):
            p += '/'
            record[p] = [t, None, count + 1]
            record = parse(v, record, p, count + 1)
        else:
            record[p] = [t, v, count]
    return record


if __name__ == '__main__':
    import json

    path = '/'.join(__file__.split('/')[:-1]) + '/settings.json'

    with open(path) as f:
        settings = json.load(f)

    s = parse(settings, current='/')
    for it in sorted(s.items(), key=lambda _: (_[1][2], _[0])):
        print(it)
