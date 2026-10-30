#!/usr/bin/env python3
# source .venv/bin/activate
# uvicorn fetch_convert:app --host 0.0.0.0 --port 8001


import re
import os
from pathlib import Path
import shutil
import subprocess
from typing import Dict
import json
import copy

import requests
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import trafilatura
import datetime
import spacy
import sys
from pathlib import Path


sys.stdout.reconfigure(line_buffering=True)

nlp={}


# Konfiguration
DEBUG=False
CHUNK_MAX = 6000
MAX_INTRO = 3000
MICRO_CHUNK_MAX = 700
MICRO_CHUNK_MIN = 100
MICRO_CHUNK_OVERLAP = 100
BASE_URL = "https://entscheidsuche.ch/docs"
DATA_DIR = Path("/var/www/files/chunks")
DATA_URL = "https://files.entscheidsuche.ch/chunks/"
DATA_DIR.mkdir(parents=True, exist_ok=True)
FILENAME_REGEX = re.compile(r'^[\w\-.]+$')  # erlaubt a-zA-Z0-9 _ - . ; verhindert Slashes und Traversal
DOPPELUMBRUCH_REGEX = re.compile(r'(?s).*\n\n', flags=re.MULTILINE)
NUMMER_REGEX = re.compile(r'(?s).*^(?P<line>[\.\dA-Za-z]{1,4})$', flags=re.MULTILINE)
WOERTER={ "de": ["der","die","das","ein","eine","einer","er","sie","ihn","hat","hatte","hätte","ist","war","sind"],"fr": ["le","lui","elle","je","on","vous","nous","leur","qui","quand","parce","que","faire","sont","vont"], "it": ["della","del","di","casi","una","al","questa","più","primo","grado","che","diritto","leggi","corte"]}
de_r=re.compile(r"\b(?:"+"|".join(WOERTER["de"])+r")\b")
fr_r=re.compile(r"\b(?:"+"|".join(WOERTER["fr"])+r")\b")
it_r=re.compile(r"\b(?:"+"|".join(WOERTER["it"])+r")\b")


REGS={'de': de_r, 'fr': fr_r, 'it': it_r}
CACHE_VALID="14.04.2026 06:00:00"
DATE_FORMAT="%d.%m.%Y %H:%M:%S"
cache_limit=datetime.datetime.strptime(CACHE_VALID,DATE_FORMAT)

app = FastAPI(title="PDF fetch & pdftotext service")


# Whitespace außer \n (Unicode-Whitespace inkl. NBSP etc., soweit Python es als whitespace klassifiziert)
_WS_NO_NL = r"[^\S\n]"

# 1) Zeilenenden / Zero-width
RE_CRLF = re.compile(r"\r\n?")
RE_ZW   = re.compile(r"[\u200B\u200C\u200D\uFEFF]")  # zero-width + BOM

# 2) Dehyphenation (restriktiv DE/FR/IT)
ALLOWED_DE_FR_IT = (
	"A-Za-z"
	"ÄÖÜäöüß"
	"ÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ"
	"àâæçéèêëîïôœùûüÿ"
	"ÌÍÒÓÚìíòóú"
	"ÀÈÉÌÍÎÒÓÙÚÛ"
)

# Kleinbuchstaben-Subset (für den "right beginnt klein"-Check ohne Callback)
ALLOWED_LOWER_DE_FR_IT = (
	"a-z"
	"äöüß"
	"àâæçéèêëîïôœùûüÿ"
	"ìíòóú"
)

# Dehyphenation: left/right je >=2, right beginnt klein, kein '-' direkt davor/danach, erlaubt Einrückungen um NL
RE_DEHYPHENATE_DE_FR_IT = re.compile(
	rf"(?m)(?<!-)\b(?P<left>[{ALLOWED_DE_FR_IT}]{{2,}})-{_WS_NO_NL}*\n{_WS_NO_NL}*(?P<right>[{ALLOWED_LOWER_DE_FR_IT}][{ALLOWED_DE_FR_IT}]{{1,}})\b(?!-)"
)

# 3) Kombinierte NL/Space-Normalisierung (2 NL / 1 NL / Space) in einem Pass
RE_WS_NORM = re.compile(r"[^\S\n]*\n(?:[^\S\n]*\n)+[^\S\n]*|[^\S\n]*\n[^\S\n]*(?!\n)|[^\S\n]+")

# Zero-width removal ohne Regex (schnell)
_ZW_TRANS = str.maketrans({"\u200B": None, "\u200C": None, "\u200D": None, "\uFEFF": None})


def normalize_ws(text: str) -> str:
	"""
	Normalisiert Whitespaces:
	  1) CRLF/CR -> LF
	  2) entfernt Zero-width-Zeichen
	  3) restriktive Dehyphenation über Zeilenumbruch (DE/FR/IT), right beginnt klein
	  4) Whitespace-Normalisierung:
		 - beliebiger WS + >=2 NL + WS -> genau 2 NL
		 - beliebiger WS + 1 NL + WS  -> genau 1 NL
		 - sonstiger WS (ohne NL)	 -> genau 1 Space
	"""

	# 0) Zeilenenden normalisieren (CRLF/CR -> LF) ohne Regex
	if "\r" in text:
		text = text.replace("\r\n", "\n").replace("\r", "\n")

	# 0.5) Zero-width entfernen (ohne Regex)
	if "\u200b" in text or "\u200c" in text or "\u200d" in text or "\ufeff" in text:
		text = text.translate(_ZW_TRANS)

	# 0.7) Dehyphenation nur wenn überhaupt "-\n" vorkommt
	if "-\n" in text or "- \n" in text:
		text = RE_DEHYPHENATE_DE_FR_IT.sub(r"\g<left>\g<right>", text)

	# 1-3) NL/Space-Normalisierung in einem Regex-Pass
	if "\n" in text or "\t" in text or "\xa0" in text or "  " in text:
		def _ws_repl(m: re.Match) -> str:
			s = m.group(0)
			if "\n" in s:
				return "\n\n" if s.count("\n") > 1 else "\n"
			return " "
		text = RE_WS_NORM.sub(_ws_repl, text)

	return text

@app.on_event("startup")
def load_models():
	exclude = ["tagger","morphologizer","lemmatizer","ner","attribute_ruler", "entity_ruler", "textcat","textcat_multilabel","spancat", "transformer"]
	nlp["de"] = spacy.load("de_core_news_sm", exclude=exclude)
	nlp["fr"] = spacy.load("fr_core_news_sm", exclude=exclude)
	nlp["it"] = spacy.load("it_core_news_sm", exclude=exclude)

	for lang in nlp:
		nlp[lang].max_length = 10000000

class ConvertRequest(BaseModel):
	filename: str

def sanitize_filename(raw: str) -> str:
	name = raw.strip()
	if not FILENAME_REGEX.match(name):
		raise ValueError("Invalid filename characters")
	return name.replace("_", "/", 3)
	
	
def download_from_url(url: str, dest: Path, timeout: int = 20) -> None:
	"""
	Download a remote URL (stream) to dest. Raises FileNotFoundError on 404,
	requests.RequestException on other networking errors.
	"""
	if DEBUG:
		print(f"download_from_ul: url: {url}, dest: {dest}")

	with requests.get(url, stream=True, timeout=timeout) as resp:
		if resp.status_code == 404:
			print(f"Remote file not found: {url}")
			return False
		resp.raise_for_status()
		tmp_path = dest.with_suffix(dest.suffix + ".tmp")
		with tmp_path.open("wb") as f:
			shutil.copyfileobj(resp.raw, f)
		tmp_path.replace(dest)
	return True

def fetch_pdf_html_for_name(name: str, destPDF: Path, destHTML: Path, timeout: int = 20) -> (str, dict):
	"""
	Check for a JSON at BASE_URL/<name>.json. If present and contains
	{'PDF': {'Datei': <path>}}, use that path to download the PDF.
	Otherwise fall back to BASE_URL/<name>.pdf.

	<path> may be:
	- an absolute URL (starts with http/https) -> used as-is
	- a relative path like "subdir/file.pdf" or "/subdir/file.pdf" -> joined with BASE_URL
	- just a filename -> treated as relative to BASE_URL
	"""
	json_url = f"{BASE_URL}/{name}.json"
	if DEBUG:
		print(f"fetch_pdf_html_for_name: url: {json_url}, pdf_path: {destPDF}, html_path {destHTML}")

	resp = requests.get(json_url, timeout=timeout)


	if resp.status_code == 200:
		# try to parse JSON
		obj = resp.json()

		# navigate structure safely
		pdf_field = None
		html_field = None
		if isinstance(obj, dict):
			pdf_info = obj.get("PDF")
			if isinstance(pdf_info, dict):
				pdf_field = pdf_info.get("Datei")
			html_info = obj.get("HTML")
			if isinstance(html_info, dict):
				html_field = html_info.get("Datei")
		if pdf_field:
			# determine final URL
			pdf_field_str = str(pdf_field).strip()
			pdf_url = f"{BASE_URL}/{pdf_field_str}"
			# download from determined URL (may raise FileNotFoundError)
			if download_from_url(pdf_url, destPDF, timeout=timeout):
				return "pdf", obj
		if html_field:
			# determine final URL
			html_field_str = str(html_field).strip()
			html_url = f"{BASE_URL}/{html_field_str}"
			# download from determined URL (may raise FileNotFoundError)
			download_from_url(html_url, destHTML, timeout=timeout)
			return "html", obj
		else:
			print("weder PDF noch HTML vorhanden")
			return None, obj
	else:
		return None, None


def run_pdftotext(pdf_path: Path, txt_path: Path) -> None:
	if DEBUG:
		print(f"run_pdftotext: pdf_path: {pdf_path}, txt_path {txt_path}")
	# Ensure pdftotext is available
	if shutil.which("pdftotext") is None:
		raise EnvironmentError("pdftotext not found in PATH. Install poppler-utils / poppler.")
	# call pdftotext -layout pdf txt
	completed = subprocess.run(
		["pdftotext", "-layout", str(pdf_path), str(txt_path)],
		check=False,
		capture_output=True,
		text=True,
	)
	if completed.returncode != 0:
		raise RuntimeError(
			f"pdftotext failed (rc={completed.returncode}): {completed.stderr.strip()}"
		)

def run_htmltotext(html_path: Path, txt_path: Path) -> None:
	"""
	Liest html_path ein, extrahiert mit trafilatura Text und speichert in txt_path.
	Raises:
	  - FileNotFoundError wenn html_path nicht existiert
	  - RuntimeError bei Extraktionsfehlern
	"""
	if DEBUG:
		print(f"run_htmltotext: pdf_path: {pdf_path}, txt_path {txt_path}")

	if not html_path.exists():
		raise FileNotFoundError(f"Local HTML not found: {html_path}")

	raw = html_path.read_bytes()
	# trafilatura erwartet bytes oder str; wir geben bytes
	extracted = trafilatura.extract(raw, include_comments=False, include_tables=True)
	if not extracted:
		raise RuntimeError(f"trafilatura failed to extract meaningful text from {html_path}")
	txt_path.write_text(extracted, encoding="utf-8", errors="replace")

@app.get("/convert")
def convert_get(dokid: str = Query(..., min_length=1)):
	"""
	GET /convert?dokid=<Dateiname>
	"""

	try:
		filename = sanitize_filename(dokid)
	except ValueError as e:
		raise HTTPException(status_code=400, detail=str(e))

	pdf_path = DATA_DIR / f"{filename}.pdf"
	html_path = DATA_DIR / f"{filename}.html"
	txt_path = DATA_DIR / f"{filename}.txt"
	json_path = DATA_DIR / f"{filename}.json"


	obj=get_json(dokid)
	scrapetime = datetime.datetime.strptime(obj["Zeit UTC"], DATE_FORMAT).replace(tzinfo=datetime.timezone.utc).astimezone().replace(tzinfo=None)
	
	#Gibt es bereits ein json mit Chunkzerlegung?
	imCache=False
	if json_path.exists():
		with json_path.open("r", encoding="utf-8") as fh:
			altobj = json.load(fh)
			zeitstring=altobj.get("Chunktime")
			if zeitstring:
				zeit=datetime.datetime.strptime(zeitstring,DATE_FORMAT)
				if zeit>cache_limit and zeit>scrapetime:
					obj=altobj
					print(f"{dokid} ist im Cache")
					imCache=True
	
	if not imCache:
		try:
			path = Path(txt_path)
			path.parent.mkdir(parents=True, exist_ok=True)
			result, obj=fetch_pdf_html_for_name(dokid, pdf_path, html_path)
			ok=False
			if result=="pdf":
				ok=run_pdftotext(pdf_path, txt_path)
			if result=="html":
				run_htmltotext(html_path, txt_path)
		except Exception as e:
			raise HTTPException(status_code=500, detail=str(e))

		if not txt_path.exists():
			raise HTTPException(status_code=500, detail=f"Text output not created {txt_path}")

		text = txt_path.read_text(encoding="utf-8", errors="replace")
		if DEBUG:
			print(f"{dokid}: Textlänge {len(text)}")
		lang=get_lang(text)
		intro=get_intro(obj,lang)
		chunk_size=CHUNK_MAX-len(intro)-2
		chunks=get_chunks(text,chunk_size, dokid, lang, obj, filename, intro)
		micro_chunks=get_micro_chunks(text,lang,dokid, filename)
		obj['Sprache']=lang
		obj["Chunktime"]=datetime.datetime.now().strftime(DATE_FORMAT)
		if chunks:
			last=0
			for c in chunks:
				c['MicroChunks']=[]
				while last<len(micro_chunks) and micro_chunks[last]['offset']<c['offset']+c['len']:
					if micro_chunks[last]['offset']+micro_chunks[last]['len']>c['offset']:
						c['MicroChunks'].append(micro_chunks[last])
					last+=1
				last-=1
			obj["Chunks"]=chunks
			with json_path.open("w", encoding="utf-8") as f:
				json.dump(obj, f, ensure_ascii=False, indent=2)
		
	return obj

def get_intro(obj, lang) -> str:
	intro=""
	kopfzeile=""
	kopf=obj.get("Kopfzeile")
	if kopf:
		for k in kopf:
			sprachen=k.get("Sprachen")
			if sprachen and lang in sprachen:
				kopfzeile=k.get("Text")
				if DEBUG:
					print(f"Kopfzeile: {kopfzeile}")

	abstrakt=""
	abstr=obj.get("Abstract")
	if abstr:
		for a in abstr:
			sprachen=a.get("Sprachen")
			if sprachen and lang in sprachen:
				abstrakt=a.get("Text")
				if DEBUG:
					print(f"Abstract: {abstrakt}")

	if lang=="de":
		intro="Auszug aus dem Entscheid "+kopfzeile
		if abstrakt:
			intro+="\nRegeste:\n"+abstrakt
	if lang=="fr":
		intro="Extrait de l'arrêt "+kopfzeile
		if abstrakt:
			intro+="\nRegeste:\n"+abstrakt
	if lang=="it":
		intro="Estratto della sentenza "+kopfzeile
		if abstrakt:
			intro+="\nRegesto:\n"+abstrakt
	if len(intro)>MAX_INTRO:
		print(f"Intro gekürzt da {len(intro)} Zeichen lang. {intro[:MAX_INTRO]} ||||| {intro[MAX_INTRO:]}")
		intro=intro[:MAX_INTRO]
	print(f"intro: {intro}")
	return intro

def write_chunk(chunk, chunknum, chunkobj, filename, intro) -> {}:
	chunkname=filename+"_C"+("000"+str(chunknum))[-4:]
	chunkurl=DATA_URL+chunkname+".json"
	chunkid={"id":chunkname, "url":chunkurl}
	chunkobj['Chunktext']=intro+"\n\n"+normalize_ws(chunk)
	chunkpath=DATA_DIR / f"{chunkname}.json"
	with chunkpath.open("w", encoding="utf-8") as f:
		json.dump(chunkobj, f, ensure_ascii=False)
	return chunkid

def write_micro_chunk(text, chunknum, filename) -> {}:
		chunk={}
		chunkid=filename+"_MC"+("0000"+str(chunknum))[-5:]
		chunkurl=DATA_URL+chunkid+".txt"
		chunk['id']=chunkid
		chunk['url']=chunkurl
		chunkpath=DATA_DIR / f"{chunkid}.txt"
		with chunkpath.open("w", encoding="utf-8") as f:
			f.write(normalize_ws(text))
		return chunk



def get_lang(text) -> str:
	lang=max(REGS, key=lambda key: len(REGS[key].findall(text)))
	return lang

def get_micro_chunks(text,lang, dokid, filename) -> ([], str):
	if DEBUG:
		print(f"{dokid}: get Micro Chunks called")
	chunks=[]
	sents=nlp[lang](text).sents
	if DEBUG:
		print(f"{dokid}: Habe sent aufgerufen.")
	saetze=list(sents)
	if DEBUG:
		print(f"{dokid}: Zerlegung des {len(text)} Zeichen langen Textes in {len(saetze)} Sätze gefunden.")
		print(f"{dokid}: Letzter Satz startet {saetze[len(saetze)-1].text}: '{saetze[len(saetze)-1].text}'")
	
	satznum=0
	offset=0
	chunks=[]
	chunknum=1
	
	while len(text)>MICRO_CHUNK_MAX:
		if DEBUG:
			print(f"{dokid}: Micro Chunk Nummer {len(chunks)+1}: {text[:50]}…")
		zerlegungstext=text[:MICRO_CHUNK_MAX]
		doppelumbruch=DOPPELUMBRUCH_REGEX.search(zerlegungstext)
		nummer=NUMMER_REGEX.search(zerlegungstext)
		pos=0
		# Es gibt einen klaren Umbruch
		if doppelumbruch:
			doppelpos=doppelumbruch.end()
			if DEBUG:
				print(f"{dokid}: Doppelpos {doppelpos}:")
			pos=doppelpos
			# Gibt es auch eine mutmassliche Nummer?
			if nummer:
				nummerpos=nummer.start('line')
				if DEBUG:
					print(f"{dokid}: Nummerpos {nummerpos}:")	
				# Ist die Nummer am Doppelumbruch, dann nehmen
				if abs(doppelpos-nummerpos)<3:
					pos=nummerpos
		else:
			if nummer:
				nummerpos=nummer.start('line')
				pos=nummerpos
		if pos>=MICRO_CHUNK_MIN:
			#Zerlegung gefunden
			if DEBUG:
				print(f"{dokid}: Absatzzerlegung gefunden an Position {pos}")
			while satznum<len(saetze) and saetze[satznum].start_char<pos+offset:
				if DEBUG:
					print(f"{dokid}: satznum {satznum}, satzpos {saetze[satznum].start_char}, pos {pos}, offset {offset}")
				satznum+=1
			endpos=pos	
		else:
			if DEBUG:
				print(f"{dokid}: Keine Absatzzerlegung gefunden, daher nach Satzzerlegung gehen")
			while satznum < len(saetze) and saetze[satznum].start_char <= offset:
				satznum += 1
			endsatznum=satznum
			while endsatznum+1<len(saetze) and saetze[endsatznum+1].start_char<offset+MICRO_CHUNK_MAX:
				endsatznum+=1
				if DEBUG:
					print(f"{dokid}: Satziteration endsatznum {endsatznum}, offset {offset}, satzpos {saetze[endsatznum].start_char}")
			# weitere Sätze gefunden
			if endsatznum > satznum:
				if DEBUG:
					print(f"{dokid}: endsatznum {endsatznum} grösser als satznum {satznum}")
				#Kurze Sätze (ggf. Überschriften, Nummerierungen) auch beim Nächsten lassen, wenn sie nicht das Einzige sind.
				while endsatznum>satznum+2 and saetze[endsatznum-1].end_char-saetze[endsatznum-1].start_char<10:
					if DEBUG:
						print(f"{dokid}: endsatznum reduziert weil len(saetze[{endsatznum-1}])={saetze[endsatznum-1].end_char-saetze[endsatznum-1].start_char}")
					endsatznum-=1
				# Letzter Satz überlappen wenn nicht lediglich ein Satz
				endpos=saetze[endsatznum].start_char-offset
				if DEBUG:
					print(f"{dokid}: endsatznum {endsatznum}, offset {offset}, satzpos[{endsatznum}] {saetze[endsatznum].start_char}, satzpos[{endsatznum-1}] {saetze[endsatznum-1].start_char}")
				if saetze[endsatznum-1].start_char>offset:
					pos=saetze[endsatznum-1].start_char-offset
				else:
					pos=endpos
				if DEBUG:
					print(f"{dokid}: endpos {endpos}, pos {pos}")
			else:
				# Anderes Trennungskriterium suchen, notfalls brute force
				endpos=0
				seitenumbruch=zerlegungstext.rfind('\x0c')
				if seitenumbruch>0:
					if DEBUG:
						print(f"Seitenumbruch bei Position {seitenumbruch} gefunden")
					endpos=seitenumbruch
				else:
					zeilenumbruch=zerlegungstext.rfind('\x0a')
					if zeilenumbruch >0:
						if DEBUG:
							print(f"Zeilenumbruch bei Position {zeilenumbruch} gefunden")
						endpos=zeilenumbruch
					else:
						leerzeichen=zerlegungstext.rfind(' ')
						if leerzeichen >0:
							if DEBUG:
								print(f"Leerzeichen bei Position {seitenumbruch} gefunden")
							endpos=leerzeichen
						else:
							endpos=len(zerlegungstext)-1
				pos=endpos
				if DEBUG:
					print(f"{dokid} Brute Force endpos {endpos} bei Zerlegungstextlänge {len(zerlegungstext)+1} und Textlänge {len(text)} sowie Zerlegunungstext: {zerlegungstext}")
				
		#Verbleibender Rest zu klein, dann nicht mehr abschneiden, letzter Chunk wird dann automatisch angehängt
		if len(text)-pos < MICRO_CHUNK_MIN:
			pos=len(text)
			break
		else:
			if pos <=0:
				print(f"Fehler A! pos={pos}, enpos={endpos}, len(text)={len(text)}, offset={offset}")
				if len(text) <= MICRO_CHUNK_MAX:
					break
				else:
					pos=MICRO_CHUNK_MAX-MICRO_CHUNK_OVERLAP
					endpos=MICRO_CHUNK_MAX
			if endpos<pos:
				print(f"Fehler B! pos={pos}, enpos={endpos}, len(text)={len(text)}, offset={offset}")
				endpos=pos
			chunk=write_micro_chunk(text[:endpos], chunknum, filename)
			chunknum+=1
			chunk['offset']=offset
			chunk['len']=len(text[:endpos])
			text=text[pos:]
			offset+=pos
			chunks.append(chunk)
	chunk=write_micro_chunk(text, chunknum, filename)
	chunk['offset']=offset
	chunk['len']=len(text)
	chunks.append(chunk)
	return chunks

def get_chunks(text, chunk_size, dokid, lang, obj, filename, intro) -> list:
	chunkobj = copy.deepcopy(obj)   # ok, einmal
	chunknum = 1
	chunks = []

	n = len(text)
	start = 0

	while (n - start) > chunk_size:
		if DEBUG:
			print(f"{dokid}: Chunk {chunknum} rest={n-start} chunk_size={chunk_size}: {text[start:start+50]}…")

		zerlegungstext = text[start:start+chunk_size]
		doppelumbruch = DOPPELUMBRUCH_REGEX.search(zerlegungstext)
		nummer = NUMMER_REGEX.search(zerlegungstext)
		pos = 0

		if doppelumbruch:
			doppelpos = doppelumbruch.end()
			pos = doppelpos
			if nummer:
				nummerpos = nummer.start("line")
				if abs(doppelpos - nummerpos) < 3:
					pos = nummerpos
		else:
			if nummer:
				pos = nummer.start("line")

		if pos <= 0:
			pos = zerlegungstext.rfind("\x0c")
			if pos <= 0:
				pos = zerlegungstext.rfind("\n")
				if pos <= 0:
					pos = zerlegungstext.rfind(" ")
					if pos <= 0:
						pos = chunk_size - 1

		chunktext = text[start:start+pos]

		chunk={}
		chunk=write_chunk(chunktext, chunknum, chunkobj, filename, intro)
		chunk['offset']=start
		chunk['len']=pos
		chunks.append(chunk)
		chunknum += 1
		start += pos

	# Rest korrekt schreiben:
	chunk={}
	chunk=write_chunk(text[start:], chunknum, chunkobj, filename, intro)
	chunk['offset']=start
	chunk['len']=len(text)-start
	chunks.append(chunk)
	return chunks

def get_json(name) -> dict:
	json_url = f"{BASE_URL}/{name}.json"
	resp = requests.get(json_url, timeout=20)

	if resp.status_code == 200:
		# try to parse JSON
		obj = resp.json()
		return obj
	else:
		return None
