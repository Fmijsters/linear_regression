import urllib.request
from urllib.request import FancyURLopener


class FixFancyURLOpener(FancyURLopener):

    def http_error_default(self, url, fp, errcode, errmsg, headers):
        if errcode == 403:
            raise ValueError("403")
        return super(FixFancyURLOpener, self).http_error_default(
            url, fp, errcode, errmsg, headers
        )

# Monkey Patch
urllib.request.FancyURLopener = FixFancyURLOpener

url = "https://www.whoscored.com/Teams/26/Archive/England-Liverpool"
urllib.request.urlretrieve(url, "cite0.bib")
