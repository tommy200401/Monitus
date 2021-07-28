from .base import FMPApi


class ProfileApi(FMPApi):
    expire = dict(minutes=2)
    endpoint = '/api/v3/profile/%(ticker)s?apikey=%(api_key)s'


class EarningCallTranscriptApi(FMPApi):
    expire = dict(minutes=2)
    endpoint = '/api/v3/earning_call_transcript/%(ticker)s?quarter=%(quarter)s&year=%(year)s&apikey=%(api_key)s'


if __name__ == '__main__':
    print(ProfileApi.get(ticker='AAPL'))
    print(EarningCallTranscriptApi.get(use_cache=False, ticker='AAPL', quarter=2, year=2019))
