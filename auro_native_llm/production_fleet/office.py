"""Dependency-free native deliverable engine for MD, CSV, DOCX, XLSX, and PDF."""
from __future__ import annotations
import csv, hashlib, json, re
from pathlib import Path
from zipfile import ZIP_DEFLATED,ZipFile
from xml.sax.saxutils import escape

class NativeOffice:
    def create_bundle(self,out_dir,title,sections,table=None):
        root=Path(out_dir); root.mkdir(parents=True,exist_ok=True); safe=_safe(title); table=table or []
        files=[]
        md="# "+title+"\n\n"+"\n\n".join("## "+str(s["heading"])+"\n\n"+str(s["body"]) for s in sections)+"\n"
        files.append(_write(root/(safe+".md"),md.encode()))
        csv_path=root/(safe+".csv")
        with csv_path.open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f); [w.writerow([str(c) for c in row]) for row in table]
        files.append(_record(csv_path))
        docx=root/(safe+".docx"); _docx(docx,title,sections); files.append(_record(docx))
        xlsx=root/(safe+".xlsx"); _xlsx(xlsx,table); files.append(_record(xlsx))
        pdf=root/(safe+".pdf"); _pdf(pdf,title,sections); files.append(_record(pdf))
        manifest={"schema":"auro.office.bundle.v1","title":title,"files":files,"formats":["md","csv","docx","xlsx","pdf"]}
        m=root/(safe+".manifest.json"); m.write_text(json.dumps(manifest,indent=2),encoding="utf-8"); manifest["manifest"]=_record(m); return manifest
def _docx(path,title,sections):
    paras=[title]+[x for s in sections for x in (str(s["heading"]),str(s["body"]))]
    body="".join("<w:p><w:r><w:t xml:space=\"preserve\">"+escape(p)+"</w:t></w:r></w:p>" for p in paras)
    with ZipFile(path,"w",ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
        z.writestr("_rels/.rels",'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
        z.writestr("word/document.xml",'<?xml version="1.0"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>'+body+'<w:sectPr/></w:body></w:document>')
def _xlsx(path,rows):
    xml_rows=[]
    for ri,row in enumerate(rows,1):
        cells="".join(f'<c r="{_col(ci)}{ri}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>' for ci,v in enumerate(row,1)); xml_rows.append(f'<row r="{ri}">{cells}</row>')
    with ZipFile(path,"w",ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",'<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/><Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/></Types>')
        z.writestr("_rels/.rels",'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        z.writestr("xl/workbook.xml",'<?xml version="1.0"?><workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Auro" sheetId="1" r:id="rId1"/></sheets></workbook>')
        z.writestr("xl/_rels/workbook.xml.rels",'<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/></Relationships>')
        z.writestr("xl/worksheets/sheet1.xml",'<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+"".join(xml_rows)+"</sheetData></worksheet>")
def _pdf(path,title,sections):
    lines=[title]+[str(x) for s in sections for x in (s["heading"],s["body"])]
    text="BT /F1 12 Tf 54 760 Td "+" ".join("("+_pdf_escape(x[:100])+f") Tj 0 -18 Td" for x in lines[:35])+" ET"
    objs=["<< /Type /Catalog /Pages 2 0 R >>","<< /Type /Pages /Kids [3 0 R] /Count 1 >>","<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",f"<< /Length {len(text.encode())} >>\nstream\n{text}\nendstream","<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    data=b"%PDF-1.4\n"; offsets=[]
    for i,o in enumerate(objs,1): offsets.append(len(data)); data+=f"{i} 0 obj\n{o}\nendobj\n".encode()
    x=len(data); data+=f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()+b"".join(f"{n:010d} 00000 n \n".encode() for n in offsets)+f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{x}\n%%EOF\n".encode(); path.write_bytes(data)
def _safe(s): return re.sub(r"[^A-Za-z0-9._-]+","-",s).strip("-") or "deliverable"
def _col(n):
    out=""
    while n: n,r=divmod(n-1,26); out=chr(65+r)+out
    return out
def _pdf_escape(s): return str(s).replace("\\","\\\\").replace("(","\\(").replace(")","\\)").replace("\n"," ")
def _write(path,data): path.write_bytes(data); return _record(path)
def _record(path):
    b=path.read_bytes(); return {"path":str(path),"name":path.name,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()}

