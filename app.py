import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import io

# ─────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaPredict | Su Kalitesi Analizi",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0a0e27 0%, #0d1b3e 50%, #0a1628 100%); }

    /* Hero başlık */
    .hero-title {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .hero-title h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7, #00d4ff);
        background-size: 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s infinite;
        margin-bottom: 0.3rem;
    }
    @keyframes shimmer { 0%{background-position:0%} 50%{background-position:100%} 100%{background-position:0%} }
    .hero-title p { color: #8898aa; font-size: 1.1rem; }

    /* Parametre kartları */
    .param-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s;
    }
    .param-card:hover { border-color: rgba(0,212,255,0.3); }
    .param-label { color: #e0e6ff; font-weight: 600; font-size: 0.95rem; }
    .who-note { color: #559aff; font-size: 0.78rem; margin-top: 0.15rem; }

    /* Sonuç kartları */
    .result-safe {
        background: linear-gradient(135deg, rgba(0,200,100,0.15), rgba(0,200,100,0.05));
        border: 2px solid rgba(0,200,100,0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: pulse-green 2s infinite;
    }
    @keyframes pulse-green {
        0%,100%{box-shadow:0 0 20px rgba(0,200,100,0.2)}
        50%{box-shadow:0 0 40px rgba(0,200,100,0.5)}
    }
    .result-danger {
        background: linear-gradient(135deg, rgba(255,60,60,0.15), rgba(255,60,60,0.05));
        border: 2px solid rgba(255,60,60,0.5);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%,100%{box-shadow:0 0 20px rgba(255,60,60,0.2)}
        50%{box-shadow:0 0 40px rgba(255,60,60,0.5)}
    }
    .result-title { font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }
    .result-sub { font-size: 1rem; color: #aab4c8; }

    /* Uyarı kutusu */
    .warn-box {
        background: rgba(255,180,0,0.1);
        border-left: 4px solid #ffb400;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        color: #ffcc55;
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }

    /* Sekme stilleri */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8898aa;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff22, #7b2ff722) !important;
        color: #00d4ff !important;
    }

    /* Buton */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        width: 100%;
        transition: opacity 0.2s, transform 0.1s;
    }
    .stButton > button:hover { opacity: 0.88; transform: translateY(-1px); }

    /* Slider rengi */
    .stSlider [data-testid="stSlider"] > div > div > div > div {
        background: linear-gradient(135deg, #00d4ff, #7b2ff7);
    }

    /* Metrik kutuları */
    [data-testid="metric-container"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# MODEL & SCALER YÜKLEMESİ
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ─────────────────────────────────────────────────────────
# WHO STANDART TANIMI
# ─────────────────────────────────────────────────────────
FEATURES = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]

WHO_INFO = {
    "ph":             {"label":"pH",                "unit":"",         "min":0.0,   "max":14.0,     "step":0.1,  "default":7.0,
                       "who":"WHO ideal: 6.5 – 8.5", "warn_low":4.0, "warn_high":10.5},
    "Hardness":       {"label":"Sertlik (Hardness)", "unit":"mg/L",    "min":47.0,  "max":323.0,    "step":0.5,  "default":196.0,
                       "who":"WHO sınır: ≤ 500 mg/L", "warn_low":50.0, "warn_high":320.0},
    "Solids":         {"label":"Toplam Çözünmüş Katı (TDS)","unit":"ppm","min":320.0,"max":61230.0,"step":10.0,"default":21000.0,
                       "who":"WHO ideal: ≤ 500 ppm, kabul: ≤ 1000 ppm", "warn_low":None, "warn_high":55000.0},
    "Chloramines":    {"label":"Kloramin",           "unit":"ppm",     "min":0.35,  "max":13.1,     "step":0.05, "default":7.1,
                       "who":"WHO güvenli sınır: ≤ 4 ppm", "warn_low":None, "warn_high":12.0},
    "Sulfate":        {"label":"Sülfat",             "unit":"mg/L",    "min":129.0, "max":481.0,    "step":0.5,  "default":333.0,
                       "who":"WHO sınır: ≤ 250 mg/L (estetik)", "warn_low":None, "warn_high":470.0},
    "Conductivity":   {"label":"İletkenlik (Conductivity)","unit":"μS/cm","min":181.0,"max":753.0, "step":0.5,  "default":421.0,
                       "who":"WHO ideal: ≤ 400 μS/cm", "warn_low":None, "warn_high":740.0},
    "Organic_carbon": {"label":"Organik Karbon (TOC)","unit":"ppm",   "min":2.2,   "max":28.3,     "step":0.1,  "default":14.0,
                       "who":"WHO sınır: ≤ 2 ppm (içme suyu)", "warn_low":None, "warn_high":26.0},
    "Trihalomethanes":{"label":"Trihalometan (THM)", "unit":"μg/L",   "min":0.738, "max":124.0,    "step":0.1,  "default":66.0,
                       "who":"WHO sınır: ≤ 100 μg/L", "warn_low":None, "warn_high":118.0},
    "Turbidity":      {"label":"Bulanıklık (Turbidity)","unit":"NTU", "min":1.45,  "max":6.74,     "step":0.01, "default":3.96,
                       "who":"WHO ideal: ≤ 1 NTU, sınır: ≤ 5 NTU", "warn_low":None, "warn_high":6.2},
}

# Veri seti ortalamaları (ideal referans profil)
IDEAL_MEANS = {
    "ph": 7.08, "Hardness": 196.4, "Solids": 22014.0,
    "Chloramines": 7.12, "Sulfate": 333.8,
    "Conductivity": 426.2, "Organic_carbon": 14.3,
    "Trihalomethanes": 66.4, "Turbidity": 3.97,
}

# ─────────────────────────────────────────────────────────
# HERO BAŞLIK
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">
    <h1>💧 AquaPredict</h1>
    <p>Yapay Zeka Destekli Su Kalitesi ve İçilebilirlik Analiz Sistemi</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SEKMELER
# ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Manuel Analiz",
    "📊 Tahmin Sonucu",
    "🧠 Model Açıklaması (XAI)",
    "📁 Toplu Analiz (CSV)"
])

# ═══════════════════════════════════════════════════════════
# TAB 1 — MANUEL VERİ GİRİŞİ
# ═══════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 🔬 Su Parametrelerini Girin")
    st.markdown("Her slider'ı kullanarak su örneğinizin ölçüm değerlerini ayarlayın.")
    st.divider()

    user_values = {}
    warnings_found = []

    cols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        info = WHO_INFO[feat]
        col  = cols[i % 3]
        with col:
            label_html = f'<div class="param-card">'
            col.markdown(label_html, unsafe_allow_html=True)

            val = col.slider(
                label=f"**{info['label']}** {'(' + info['unit'] + ')' if info['unit'] else ''}",
                min_value=float(info["min"]),
                max_value=float(info["max"]),
                value=float(info["default"]),
                step=float(info["step"]),
                key=f"slider_{feat}",
            )
            col.markdown(f'<div class="who-note">🌍 {info["who"]}</div>', unsafe_allow_html=True)
            col.markdown('</div>', unsafe_allow_html=True)

            user_values[feat] = val

            # Uc değer kontrolü
            if info["warn_high"] and val >= info["warn_high"]:
                warnings_found.append(f"⚠️ **{info['label']}** çok yüksek ({val:.2f} {info['unit']}) — lütfen değeri kontrol edin.")
            if info["warn_low"]  and val <= info["warn_low"]:
                warnings_found.append(f"⚠️ **{info['label']}** çok düşük ({val:.2f}) — bu uç bir değerdir.")

    st.divider()

    if warnings_found:
        st.markdown("#### 🚨 Anlık Uyarılar")
        for w in warnings_found:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)
        st.markdown("")

    analyze_btn = st.button("🔍 Analiz Et", key="analyze_main")

    if analyze_btn:
        with st.spinner("Model tahmin yapıyor..."):
            input_df  = pd.DataFrame([user_values])
            scaled    = scaler.transform(input_df)
            pred      = model.predict(scaled)[0]
            prob      = model.predict_proba(scaled)[0]

            st.session_state["prediction"]   = int(pred)
            st.session_state["probability"]  = float(prob[1])   # içilebilir olasılığı
            st.session_state["user_values"]  = user_values
            st.session_state["analyzed"]     = True

        st.success("✅ Analiz tamamlandı! **'Tahmin Sonucu'** sekmesine geçin.")


# ═══════════════════════════════════════════════════════════
# TAB 2 — SONUÇ PANELİ
# ═══════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.get("analyzed"):
        st.info("ℹ️ Henüz analiz yapılmadı. **'Manuel Analiz'** sekmesinden verileri girip 'Analiz Et' butonuna basın.")
        st.stop()

    pred  = st.session_state["prediction"]
    prob  = st.session_state["probability"]   # P(içilebilir)

    prob_display   = prob * 100
    prob_display_r = (1 - prob) * 100

    st.markdown("### 📊 Tahmin Sonucu")
    st.divider()

    # Sonuç kartı
    if pred == 1:
        st.markdown("""
        <div class="result-safe">
            <div class="result-title">✅ GÜVENLİ İÇME SUYU</div>
            <div class="result-sub">Model bu suyu <strong>içilebilir</strong> olarak sınıflandırdı.</div>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="result-danger">
            <div class="result-title">❌ RİSKLİ SU</div>
            <div class="result-sub">Model bu suyu <strong>içilemez</strong> olarak sınıflandırdı.<br>
            Filtreleme veya arıtma önerilir.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Metrik satırı
    m1, m2, m3 = st.columns(3)
    m1.metric("🟢 İçilebilir Olasılığı",  f"%{prob_display:.1f}")
    m2.metric("🔴 İçilemez Olasılığı",     f"%{prob_display_r:.1f}")
    m3.metric("🤖 Model Kararı", "İçilebilir ✅" if pred == 1 else "İçilemez ❌")

    st.divider()
    st.markdown("#### ⚡ Olasılık Göstergesi (Gauge Chart)")

    # Gauge rengi
    gauge_color = "#00c864" if pred == 1 else "#ff3c3c"

    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_display,
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": "#00c864"}, "decreasing": {"color": "#ff3c3c"}},
        number={"suffix": "%", "font": {"size": 42, "color": "#e0e6ff"}},
        title={"text": "İçilebilir Olasılığı", "font": {"size": 18, "color": "#8898aa"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334",
                     "tickfont": {"color": "#8898aa"}},
            "bar":  {"color": gauge_color, "thickness": 0.35},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "rgba(255,60,60,0.15)"},
                {"range": [40, 60], "color": "rgba(255,180,0,0.1)"},
                {"range": [60,100], "color": "rgba(0,200,100,0.1)"},
            ],
            "threshold": {
                "line":  {"color": "#ffb400", "width": 3},
                "thickness": 0.85,
                "value": 50,
            },
        }
    ))
    gauge_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font={"family": "Inter"},
        height=350,
        margin={"t": 60, "b": 20, "l": 40, "r": 40},
    )
    st.plotly_chart(gauge_fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 3 — XAI (AÇIKLANABILIR YZ)
# ═══════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🧠 Model Neden Bu Kararı Verdi?")
    st.markdown("Model'in karar mekanizmasını şeffaf hale getiren iki grafik aşağıdadır.")
    st.divider()

    col_a, col_b = st.columns([1, 1])

    # ── Feature Importance ──────────────────────────────────
    with col_a:
        st.markdown("#### 🏆 Özellik Önem Sıralaması")
        st.caption("Random Forest modelinin karar verirken her parametreye verdiği önem ağırlığı")

        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            "Parametre": [WHO_INFO[f]["label"] for f in FEATURES],
            "Önem":      importances
        }).sort_values("Önem")

        max_imp = feat_df["Önem"].max()
        colors  = [
            f"rgba({int(123 + (0-123)*(v/max_imp))}, "
            f"{int(47  + (212-47)*(v/max_imp))}, "
            f"{int(247 + (255-247)*(v/max_imp))}, 0.85)"
            for v in feat_df["Önem"]
        ]

        bar_fig = go.Figure(go.Bar(
            x=feat_df["Önem"],
            y=feat_df["Parametre"],
            orientation="h",
            marker={"color": colors, "line": {"color": "rgba(0,212,255,0.3)", "width": 1}},
            text=[f"{v:.3f}" for v in feat_df["Önem"]],
            textposition="outside",
            textfont={"color": "#e0e6ff", "size": 11},
        ))
        bar_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor ="rgba(255,255,255,0.02)",
            font={"color": "#8898aa", "family": "Inter"},
            xaxis={"title": "Önem Skoru", "gridcolor": "rgba(255,255,255,0.06)"},
            yaxis={"title": ""},
            height=420,
            margin={"t": 20, "b": 40, "l": 10, "r": 60},
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # ── Radar Chart ─────────────────────────────────────────
    with col_b:
        st.markdown("#### 🕸️ Su Profili Karşılaştırması")
        st.caption("Girdiğiniz su profili (mavi) ile veri setindeki ortalama profil (turuncu) karşılaştırması")

        if st.session_state.get("user_values"):
            uv = st.session_state["user_values"]

            # Min-max normalize (0-1 aralığı için)
            def norm(feat, val):
                mn = WHO_INFO[feat]["min"]
                mx = WHO_INFO[feat]["max"]
                return (val - mn) / (mx - mn) if mx != mn else 0.5

            labels = [WHO_INFO[f]["label"] for f in FEATURES]
            user_r  = [norm(f, uv[f])              for f in FEATURES]
            ideal_r = [norm(f, IDEAL_MEANS[f])     for f in FEATURES]

            # Kapatma için ilk değeri sona ekle
            labels_c = labels + [labels[0]]
            user_rc  = user_r + [user_r[0]]
            ideal_rc = ideal_r + [ideal_r[0]]

            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=ideal_rc, theta=labels_c,
                fill="toself",
                fillcolor="rgba(255,165,0,0.12)",
                line={"color": "#ffb400", "width": 2},
                name="Ortalama Profil",
            ))
            radar_fig.add_trace(go.Scatterpolar(
                r=user_rc, theta=labels_c,
                fill="toself",
                fillcolor="rgba(0,212,255,0.12)",
                line={"color": "#00d4ff", "width": 2.5},
                name="Girdiğiniz Su",
            ))
            radar_fig.update_layout(
                polar={
                    "bgcolor": "rgba(0,0,0,0)",
                    "radialaxis": {"visible": True, "range": [0, 1],
                                   "gridcolor": "rgba(255,255,255,0.08)",
                                   "tickfont": {"color": "#556"}},
                    "angularaxis": {"gridcolor": "rgba(255,255,255,0.08)",
                                    "tickfont": {"color": "#8898aa", "size": 10}},
                },
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "color": "#8898aa"},
                legend={"font": {"color": "#e0e6ff"}, "bgcolor": "rgba(0,0,0,0)"},
                height=420,
                margin={"t": 20, "b": 40, "l": 40, "r": 40},
            )
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Radar grafiği için önce **'Manuel Analiz'** sekmesinden verilerini girin ve analiz et.")


# ═══════════════════════════════════════════════════════════
# TAB 4 — TOPLU ANALİZ
# ═══════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📁 Toplu CSV Analizi")
    st.markdown(
        "Yüzlerce su örneğini tek seferde analiz edin. "
        "CSV dosyanız **9 sütun** içermelidir: "
        "`ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity`"
    )
    st.divider()

    # Örnek CSV indirme
    sample_data = pd.DataFrame([
        {f: WHO_INFO[f]["default"] for f in FEATURES},
        {f: WHO_INFO[f]["min"] * 1.1 for f in FEATURES},
    ])
    sample_csv = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Örnek CSV Şablonunu İndir",
        data=sample_csv,
        file_name="aquapredict_sample.csv",
        mime="text/csv",
    )

    st.divider()
    uploaded = st.file_uploader(
        "CSV dosyanızı sürükleyip bırakın veya seçin",
        type=["csv"],
        key="bulk_upload",
    )

    if uploaded:
        try:
            df_bulk = pd.read_csv(uploaded)

            # Sütun kontrolü
            missing_cols = [c for c in FEATURES if c not in df_bulk.columns]
            if missing_cols:
                st.error(f"❌ Eksik sütunlar: {', '.join(missing_cols)}")
                st.stop()

            st.markdown(f"**{len(df_bulk)} adet su örneği yüklendi.** Önizleme:")
            st.dataframe(df_bulk.head(5), use_container_width=True)

            if st.button("🚀 Tümünü Analiz Et", key="bulk_analyze"):
                with st.spinner(f"{len(df_bulk)} örnek analiz ediliyor..."):
                    X_bulk    = df_bulk[FEATURES].copy()
                    X_scaled  = scaler.transform(X_bulk)
                    preds     = model.predict(X_scaled)
                    probas    = model.predict_proba(X_scaled)[:, 1]

                    df_bulk["Tahmin (İçilebilir=1)"] = preds
                    df_bulk["İçilebilir Olasılığı (%)"] = (probas * 100).round(2)
                    df_bulk["Sonuç"] = df_bulk["Tahmin (İçilebilir=1)"].map(
                        {1: "✅ GÜVENLİ", 0: "❌ RİSKLİ"}
                    )

                # Özet metrikler
                safe_count   = int((preds == 1).sum())
                danger_count = int((preds == 0).sum())
                avg_prob     = probas.mean() * 100

                s1, s2, s3 = st.columns(3)
                s1.metric("✅ İçilebilir Örnek",  f"{safe_count}")
                s2.metric("❌ Riskli Örnek",       f"{danger_count}")
                s3.metric("📊 Ort. İçilebilirlik", f"%{avg_prob:.1f}")

                st.markdown("#### 📋 Sonuç Tablosu")
                st.dataframe(df_bulk, use_container_width=True)

                # Excel çıktısı
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine="openpyxl") as writer:
                    df_bulk.to_excel(writer, index=False, sheet_name="AquaPredict Sonuçları")
                excel_bytes = output.getvalue()

                st.download_button(
                    label="📥 Excel Raporu İndir (.xlsx)",
                    data=excel_bytes,
                    file_name="aquapredict_rapor.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        except Exception as e:
            st.error(f"❌ Hata: {e}")

# ─────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<div style="text-align:center;color:#334;font-size:0.8rem;">'
    "AquaPredict · Random Forest Modeli · WHO Standartları · XAI Dashboard"
    "</div>",
    unsafe_allow_html=True,
)
