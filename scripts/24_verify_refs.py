from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE

while not os.path.isdir(os.path.join(ROOT, 'src')) and os.path.dirname(ROOT) != ROOT:
    ROOT = os.path.dirname(ROOT)

BIB = os.path.join(ROOT, 'paper', 'refs.bib')
UA = {'User-Agent': 'ref-verify/1.0 (mailto:kx@example.org)'}
ARXIV = '10.48550'


def norm(s):

    s = re.sub(r"\\[a-zA-Z]?'?", '', s or '')
    s = re.sub(r'[{}]', '', s)
    s = unicodedata.normalize('NFKD', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s.lower())
    return re.sub(r'\s+', ' ', s).strip()


def parse_bib(path):
    text = open(path, encoding='utf-8').read()
    entries = []
    for m in re.finditer(r'@(\w+)\{([^,]+),(.*?)\n\}', text, re.S):
        body = m.group(3)
        fields = {}


        for f in re.finditer(r'^\s*(\w+)\s*=\s*\{(.*?)\}\s*,?\s*$', body, re.S | re.M):
            fields[f.group(1).lower()] = re.sub(r'\s+', ' ', f.group(2)).strip()
        entries.append(dict(kind=m.group(1), key=m.group(2).strip(), **fields))
    return entries


def get_json(url, timeout=45):
    req = urllib.request.Request(url, headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def crossref(doi):
    url = 'https://api.crossref.org/works/' + urllib.parse.quote(doi) + '?mailto=kx@example.org'
    try:
        return get_json(url)['message'], 'Crossref'
    except Exception as e:
        return {'__error__': str(e)}, 'Crossref'


def datacite(doi):
    url = 'https://api.datacite.org/dois/' + urllib.parse.quote(doi, safe='')
    try:
        d = get_json(url)['data']['attributes']
        return {'title': [d.get('titles', [{}])[0].get('title', '')],
                'author': [{'family': (c.get('name') or '').split(',')[0].strip()}
                           for c in (d.get('creators') or [])],
                'issued': {'date-parts': [[d.get('publicationYear')]]}}, 'DataCite'
    except Exception as e:
        return {'__error__': str(e)}, 'DataCite'


def crossref_query(title):
    url = ('https://api.crossref.org/works?rows=3&mailto=kx@example.org&query.bibliographic='
           + urllib.parse.quote(title))
    try:
        items = get_json(url)['message']['items']
    except Exception:
        return None
    for it in items:
        t = norm(it.get('title', [''])[0])
        if t[:40] and (t[:40] in norm(title) or norm(title)[:40] in t):
            return it
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    entries = parse_bib(BIB)
    print(f'[refs] {len(entries)} entries in {os.path.relpath(BIB, ROOT)}\n')
    counts = {}
    for e in entries:
        issues, level = [], 'OK'
        doi = e.get('doi')
        if doi and doi.startswith(ARXIV):
            m, src = datacite(doi)
        elif doi:
            m, src = crossref(doi)
            if '__error__' in m:
                issues.append(f'DOI does not resolve at Crossref ({m["__error__"][:40]})')
                m = crossref_query(e.get('title', '')) or {}
                src = 'Crossref bibliographic query'
        else:
            m = crossref_query(e.get('title', '')) or {}
            src = 'Crossref bibliographic query (no DOI)'
        if not m:
            level = 'UNVERIFIABLE'
            m = {}

        au = m.get('author') or []
        first_bib = (e.get('author') or '').split(' and ')[0]
        if au and first_bib and au[0].get('family'):
            if norm(au[0]['family']).split()[0] not in norm(first_bib):
                issues.append(f'first author: bib={first_bib!r} vs {au[0]["family"]!r}')
        t_src = (m.get('title') or [''])[0]
        if t_src and norm(t_src)[:40] not in norm(e.get('title', '')) \
                and norm(e.get('title', ''))[:40] not in norm(t_src):
            issues.append(f'title: {t_src[:60]!r}')
        y_src = (m.get('issued', {}).get('date-parts') or [[None]])[0][0]
        if y_src and e.get('year') and str(y_src) != e['year'].strip('{}'):


            if src == 'DataCite' and (e.get('booktitle') or e.get('journal')):
                print(f'    (info) {e["key"]}: arXiv record {y_src}, entry cites the '
                      f'{(e.get("booktitle") or e.get("journal"))[:52]} version')
            else:
                issues.append(f"year: bib={e['year']} vs {y_src}")
        if m.get('volume') and e.get('volume') and str(m['volume']) != e['volume'].strip('{}'):
            issues.append(f"volume: bib={e['volume']} vs {m['volume']}")
        if m.get('page') and e.get('pages'):
            p_bib = e['pages'].replace('--', '-').strip('{}')
            if p_bib != m['page'] and p_bib not in m['page']:
                issues.append(f"pages: bib={p_bib} vs {m['page']}")

        if issues:
            level = 'CHECK'
        counts[level] = counts.get(level, 0) + 1
        flag = ' ' if level == 'OK' else '!'
        print(f'{flag} {e["key"]:<20} {src:<32} '
              f'{y_src or "-"} | {m.get("volume") or "-"}({m.get("issue") or "-"}) '
              f'{m.get("page") or m.get("article-number") or "-"}')
        for i in issues:
            print(f'      -> {i}')
        time.sleep(0.34)
    print(f'\n{counts}')
    print('note: a finding is a warning to inspect, not an automatic correction — identifiers can '
          'be wrong in the source record too.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
