"""Bounded CSV parsing and public HTTPS fetching with pinned DNS resolution."""
import io
import ipaddress
import socket
import time
from pathlib import PurePosixPath
from urllib.parse import urlsplit, urljoin, unquote
import certifi
import numpy as np
import pandas as pd
import urllib3

MAX_BYTES=100*1024*1024
MAX_ROWS=250000
MAX_COLS=300
MAX_CELLS=5_000_000

def parse_csv(raw):
    if len(raw)>MAX_BYTES: raise ValueError('CSV files must be 100 MB or smaller.')
    if raw.lstrip().lower().startswith((b'<!doctype html',b'<html')): raise ValueError('The URL returned a web page. Use a direct link to a CSV file.')
    try:
        header=pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig',nrows=0)
        if not 0<len(header.columns)<=MAX_COLS: raise ValueError('Use at most 300 columns.')
        limit=min(MAX_ROWS,MAX_CELLS//len(header.columns))
        df=pd.read_csv(io.BytesIO(raw),encoding='utf-8-sig',nrows=limit+1)
    except (UnicodeError,pd.errors.ParserError,pd.errors.EmptyDataError) as exc:
        raise ValueError('Could not read the CSV. Use UTF-8 encoding, a header row, and comma-separated values.') from exc
    if len(df)==0: raise ValueError('The CSV needs headers and at least one data row.')
    if len(df)>limit: raise ValueError('This dataset exceeds the processing limit: 250,000 rows or 5 million cells.')
    return df.replace([np.inf,-np.inf],np.nan)

def resolve_public_url(url):
    if any(ord(c)<32 for c in url): raise ValueError('Invalid URL.')
    try:
        parts=urlsplit(url)
        if parts.scheme!='https' or not parts.hostname or parts.username or parts.password or parts.port not in (None,443):
            raise ValueError('Use a public HTTPS URL without credentials, on the standard port.')
        addresses=socket.getaddrinfo(parts.hostname,443,type=socket.SOCK_STREAM)
    except (OSError,ValueError) as exc:
        raise ValueError('Use a reachable public HTTPS URL without credentials or a custom port.') from exc
    ips=list(dict.fromkeys(item[4][0] for item in addresses))
    if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
        raise ValueError('Private, loopback, and local network URLs are not supported.')
    return parts,ips[0]

def fetch_csv(url):
    started=time.monotonic()
    for _ in range(4):
        parts,ip=resolve_public_url(url)
        # Connect to the validated address, retaining TLS certificate/SNI checks for the hostname.
        pool=urllib3.HTTPSConnectionPool(ip,port=443,assert_hostname=parts.hostname,
            server_hostname=parts.hostname,cert_reqs='CERT_REQUIRED',ca_certs=certifi.where(),
            timeout=urllib3.Timeout(connect=5,read=10),retries=False)
        response=None
        try:
            path=parts.path or '/'
            if parts.query: path+='?'+parts.query
            response=pool.urlopen('GET',path,headers={'Host':parts.hostname,'Accept':'text/csv,text/plain','Accept-Encoding':'identity'},
                preload_content=False,redirect=False,assert_same_host=False)
            if response.status in (301,302,303,307,308):
                location=response.headers.get('Location')
                if not location: raise ValueError('The dataset server returned an invalid redirect.')
                url=urljoin(url,location)
                continue
            if response.status!=200: raise ValueError(f'The dataset server returned HTTP {response.status}.')
            raw=bytearray()
            for chunk in response.stream(65536,decode_content=True):
                raw.extend(chunk)
                if len(raw)>MAX_BYTES: raise ValueError('The remote CSV exceeds 100 MB.')
                if time.monotonic()-started>60: raise ValueError('The remote download exceeded 60 seconds.')
            name=PurePosixPath(unquote(parts.path)).name or 'remote_dataset.csv'
            if not name.lower().endswith('.csv'): name+=' .csv'
            return bytes(raw),name
        except urllib3.exceptions.HTTPError as exc:
            raise ValueError('Could not fetch the CSV securely. Check that the URL is public and its certificate is valid.') from exc
        finally:
            if response: response.close()
            pool.close()
    raise ValueError('The dataset URL redirected too many times.')
