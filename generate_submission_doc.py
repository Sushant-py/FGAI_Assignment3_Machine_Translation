"""
generate_submission_doc.py
Generates the submission Word document (.docx) conforming to Step 3 instructions:
File name: YourName_RegNum_FGAI_Assignment3.docx

Uses standard Python zipfile to construct a valid Office Open XML (.docx) package
without requiring external third-party dependencies.
"""

import os
import zipfile

def create_submission_docx(
    student_name: str = "Sushant Shekhar Sinha",
    reg_number: str = "230953472",
    assignment_title: str = "Foundations of Generative AI (ICT4403) - Assignment 3",
    system_title: str = "Comparative Neural Machine Translation: Encoder-Decoder vs. Decoder-Only LLMs",
    dataset_name: str = "OPUS-Books English-French (en-fr) Parallel Corpus",
    dataset_link: str = "https://huggingface.co/datasets/opus_books",
    github_link: str = "https://github.com/Sushant-py/FGAI_Assignment3_Machine_Translation",
    output_filename: str = "Sushant_Shekhar_Sinha_230953472_FGAI_Assignment3.docx"
):
    """Generates the standardized submission Word document."""
    
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""

    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""

    doc_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
</w:styles>"""

    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <!-- Title -->
    <w:p>
      <w:pPr>
        <w:jc w:val="center"/>
        <w:spacing w:before="240" w:after="120"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:b/>
          <w:sz w:val="36"/>
          <w:color w:val="1F497D"/>
        </w:rPr>
        <w:t>Foundations of Generative AI (ICT4403)</w:t>
      </w:r>
    </w:p>

    <!-- Subtitle -->
    <w:p>
      <w:pPr>
        <w:jc w:val="center"/>
        <w:spacing w:after="360"/>
      </w:pPr>
      <w:r>
        <w:rPr>
          <w:b/>
          <w:sz w:val="28"/>
          <w:color w:val="595959"/>
        </w:rPr>
        <w:t>Assignment 3: Machine Translation Submission Report</w:t>
      </w:r>
    </w:p>

    <!-- Table of Student and Project Details -->
    <w:tbl>
      <w:tblPr>
        <w:tblW w:w="9200" w:type="dxa"/>
        <w:tblBorders>
          <w:top w:val="single" w:sz="8" w:space="0" w:color="B8CCE4"/>
          <w:bottom w:val="single" w:sz="8" w:space="0" w:color="B8CCE4"/>
          <w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/>
          <w:insideV w:val="none"/>
        </w:tblBorders>
      </w:tblPr>

      <!-- Row 1: Student Name -->
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1F497D"/></w:rPr><w:t>Student Name:</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="6200" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:sz w:val="22"/></w:rPr><w:t>{student_name}</w:t></w:r></w:p>
        </w:tc>
      </w:tr>

      <!-- Row 2: Registration Number -->
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1F497D"/></w:rPr><w:t>Registration Number:</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="6200" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:sz w:val="22"/></w:rPr><w:t>{reg_number}</w:t></w:r></w:p>
        </w:tc>
      </w:tr>

      <!-- Row 3: Assignment Title -->
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1F497D"/></w:rPr><w:t>Assignment Title:</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="6200" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:sz w:val="22"/></w:rPr><w:t>{assignment_title}</w:t></w:r></w:p>
        </w:tc>
      </w:tr>

      <!-- Row 4: Title of Developed Autoregressive System -->
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1F497D"/></w:rPr><w:t>System Title:</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="6200" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/></w:rPr><w:t>{system_title}</w:t></w:r></w:p>
        </w:tc>
      </w:tr>

      <!-- Row 5: Name and Link of Dataset -->
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1F497D"/></w:rPr><w:t>Dataset Name &amp; Link:</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="6200" w:type="dxa"/></w:tcPr>
          <w:p>
            <w:r><w:rPr><w:sz w:val="22"/></w:rPr><w:t>{dataset_name} ({dataset_link})</w:t></w:r>
          </w:p>
        </w:tc>
      </w:tr>

      <!-- Row 6: GitHub Repository Link -->
      <w:tr>
        <w:tc>
          <w:tcPr><w:tcW w:w="3000" w:type="dxa"/></w:tcPr>
          <w:p><w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="1F497D"/></w:rPr><w:t>GitHub Repository Link:</w:t></w:r></w:p>
        </w:tc>
        <w:tc>
          <w:tcPr><w:tcW w:w="6200" w:type="dxa"/></w:tcPr>
          <w:p>
            <w:r><w:rPr><w:b/><w:sz w:val="22"/><w:color w:val="002060"/></w:rPr><w:t>{github_link}</w:t></w:r>
          </w:p>
        </w:tc>
      </w:tr>
    </w:tbl>

    <w:p><w:pPr><w:spacing w:before="360"/></w:pPr></w:p>

    <!-- Brief Executive Summary -->
    <w:p>
      <w:pPr><w:spacing w:before="120" w:after="80"/></w:pPr>
      <w:r><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="1F497D"/></w:rPr><w:t>Executive Summary &amp; Results Overview</w:t></w:r>
    </w:p>
    <w:p>
      <w:r>
        <w:t>This submission delivers a rigorous comparative benchmark of four neural machine translation architectures for English-to-French translation: (1) Basic Transformer (Encoder-Decoder) trained from scratch, (2) Multilingual T5 (google/mt5-small), (3) LLaMA-3.2-1B-Instruct (Decoder-Only), and (4) Gemma-2-2B-IT (Decoder-Only).</w:t>
      </w:r>
    </w:p>
    <w:p>
      <w:r>
        <w:t>Results indicate that Gemma-2-2B achieved the highest translation quality (38.75 BLEU, 63.92 chrF), while mT5 provided superior computational efficiency with 49.33 ms inference latency. Temperature sampling analysis revealed that greedy decoding (T = 0.0) maximizes translation fidelity, whereas higher temperatures (T &gt;= 0.7) induce lexical drift and hallucinations. Full code, notebooks, test samples, and ethical analyses are available in the linked GitHub repository.</w:t>
      </w:r>
    </w:p>

    <w:sectPr>
      <w:pgSz w:w="12240" w:h="15840"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>"""

    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as docx:
        docx.writestr('[Content_Types].xml', content_types)
        docx.writestr('_rels/.rels', rels)
        docx.writestr('word/_rels/document.xml.rels', doc_rels)
        docx.writestr('word/document.xml', document_xml)
        docx.writestr('word/styles.xml', styles_xml)

    print(f"[+] Successfully generated Word document: {output_filename}")


if __name__ == "__main__":
    create_submission_docx()
