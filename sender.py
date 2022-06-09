import requests
from take_audio import recognize_wav_file


if __name__ == '__main__':
    default = '/home/hungshing/FastData/ezTalk/users/msn9110/voice_data/uploads/sentence-t2/我不想吃義大利麵/20220525-155634.wav'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, default=default, help='''/ooo/.../xxx.wav''')
    args, _ = parser.parse_known_args()
    path = args.path
    *_, res = recognize_wav_file(
        path=path, number=50, user='msn9110')
    data = {'data': res}

    r = requests.post('http://localhost:5555/construct', json=data, timeout=10.0)
    print(r.json())
