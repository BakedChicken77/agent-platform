import io

import pandas as pd
from fastapi import UploadFile

from service.files_router import sniff_mime_from_upload
from service.storage import is_allowed


def test_sniff_mime_from_upload_accepts_xlsx() -> None:
    df = pd.DataFrame({"value": [1, 2]})
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)

    upload = UploadFile(
        filename="numbers.xlsx",
        file=io.BytesIO(buffer.getvalue()),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    mime = sniff_mime_from_upload(upload)

    assert mime == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    assert is_allowed(mime)
