# Refresh the written report

The PDF is a ten-page report generated from the verified CSV and
artifacts/metadata.json. It includes the student name and coursework
repository URL. To regenerate from the project root:

    python -m pip install 'reportlab>=4.2,<5'
    python reports/build_report.py

The output is reports/MLOps_Experimental_Learning_Report.pdf. The VM run is documented in LAB_VERIFICATION.md. Screenshots and the video
link are included in the final Word submission document.
