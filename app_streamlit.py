# app_streamlit.py
import io
import json
import numpy as np
import streamlit as st
import cv2
from detector import detect_whitehair_bytes

st.set_page_config(page_title="WhiteHair Detector", layout="centered")

st.title("白髪を自動検出してJSON出力する（Streamlitデモ）")
st.write("画像をアップロードすると、白髪っぽい細線を検出して比率をJSONで返します。\
※精度はネタ品質です。照明反射やハイライト髪を拾います。")

col1, col2 = st.columns(2)
thresh_l = col1.slider("明度しきい値（LAB L）", 150, 255, 200, 1)
min_len = col2.slider("最小領域ピクセル（ノイズ除去）", 1, 200, 10, 1)
morph = st.slider("モルフォロジOPENカーネル", 0, 5, 1, 1)

uploaded = st.file_uploader("画像ファイルを選択（jpg/png）", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_bytes = uploaded.read()
    res = detect_whitehair_bytes(img_bytes, thresh_l, min_len, morph)

    # 入力と可視化の並べ表示
    nparr = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb_in = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_vis = cv2.cvtColor(res["visual"], cv2.COLOR_BGR2RGB)

    st.subheader("プレビュー")
    c1, c2 = st.columns(2)
    with c1:
        st.image(rgb_in, caption="入力画像", use_column_width=True)
    with c2:
        st.image(rgb_vis, caption="検出可視化（赤=白髪候補）", use_column_width=True)

    # JSON表示
    payload = {
        "whitehair_ratio": res["whitehair_ratio"],
        "whitehair_pixels": res["whitehair_pixels"],
        "message": "We found some truth. Stay strong."
    }
    st.subheader("JSON 出力")
    st.code(json.dumps(payload, ensure_ascii=False, indent=2), language="json")

    # ダウンロード
    st.download_button(
        "JSONをダウンロード",
        data=json.dumps(payload, ensure_ascii=False, indent=2),
        file_name="result.json",
        mime="application/json",
    )

    # 可視化画像ダウンロード
    success, png_bytes = cv2.imencode(".png", cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR))
    if success:
        st.download_button(
            "可視化画像をダウンロード",
            data=png_bytes.tobytes(),
            file_name="whitehair_visual.png",
            mime="image/png",
        )

st.markdown("---")
st.caption("技術で“老い”は測れるが、“成熟”のアルゴリズムはまだない。")

