# Report-generation notes

The script creates a local PDF from the verified CSV and artifacts/metadata.json when a PDF preview is needed. The final coursework submission is the Word document, which contains the report, screenshots, repository link, Azure API link, and video link. To regenerate the local preview from the project root:

    python -m pip install 'reportlab>=4.2,<5'
    python reports/build_report.py

The script can regenerate a local PDF report when needed. The VM run is documented in LAB_VERIFICATION.md. Live screenshots and the video link are included in the final Word submission document.
