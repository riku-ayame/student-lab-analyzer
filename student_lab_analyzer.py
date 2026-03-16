import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import base64  # 👈 ✨OCR機能のために追加！画像をテキストにする道具
from io import StringIO  # 👈 ✨OCR機能のために追加！テキストをファイルっぽく扱う道具
from scipy.signal import find_peaks, savgol_filter# 👈　✨ 機能追加：波形解析のための強力な道具をインポート！
from scipy.interpolate import griddata  # 👈 ✨ 機能追加：3D曲面を作るための空間補間ツール！
import pickle# 👇 ✨ 追加機能：プロジェクトの保存と復元のための道具
import io
# AI関連のライブラリ
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.messages import HumanMessage # 👈 ✨OCR機能のために追加！画像をAIに渡す道具


# 🔑 APIキーの設定（StreamlitのSecrets機能を使って安全に読み込む！）
# ローカル環境でもクラウド環境でも動くように、st.secretsから取得する
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ==========================================
# 🛠️ 1. ページ全体の設定
# ==========================================
st.set_page_config(page_title="最強の学生実験アナライザー", page_icon="🧪", layout="wide")
st.title("🧪 最強の学生実験アナライザー")
st.markdown("実験データの結合、クレンジング、誤差棒付きグラフ化、理論値比較、AI解析までを全自動化します。")

# ==========================================
# 🧠 2. アプリの「記憶（Session State）」
# ==========================================
#⚠️ 重要！セッションステートでデータフレームを管理しないと、タブを切り替えるたびにデータが消えてしまいます。
#st.session_state という『絶対に消えない金庫』にデータを出し入れしているイメージです。
if "raw_df_a" not in st.session_state: st.session_state.raw_df_a = None
if "raw_df_b" not in st.session_state: st.session_state.raw_df_b = None
if "working_df" not in st.session_state: st.session_state.working_df = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# 👇 ✨ 追加：1つ前の状態を保存しておく箱
if "prev_working_df" not in st.session_state: st.session_state.prev_working_df = None
# 👇 ✨ 追加：クレンジング前の「一番最初のデータ」を保存する箱
if "original_working_df" not in st.session_state: st.session_state.original_working_df = None
# 👇 ✨ 追加：エディタを強制リセットするための「鍵」
#動的キー：前に数字（変数）をつけて、リセット時に数字をカウントアップさせる
if "editor_key" not in st.session_state: st.session_state.editor_key = 0
# 👇 ✨ 追加：グラフを表示し続けるための「スイッチ」
if "show_graph" not in st.session_state: st.session_state.show_graph = False

# ==========================================
# 📱 3. サイドバー（全体設定）
# ==========================================
with st.sidebar:
    st.header("⚙️ 全体設定")
    # ✨ 機能追加：AIアシスト機能のON/OFFスイッチ！
    use_ai = st.toggle("🤖 AIアシスト機能を有効にする", value=True)
    if use_ai:
        ai_model = st.selectbox("使用するAIモデル", ["gemini-2.5-flash", "gemini-2.5-pro"])
    
    st.divider()
    st.header("📐 データとグラフの設定")
    sig_figs = st.slider("🔢 表の有効数字（小数点以下の桁数）", min_value=0, max_value=6, value=2)
    theme = st.selectbox("🎨 グラフのカラーテーマ", ["plotly", "plotly_white", "plotly_dark", "ggplot2"])

    # 👇 ✨ 追加 ▼▼▼
    st.divider()
    st.header("💾 プロジェクト管理")

    # ↩️ Undo（元に戻す）ボタン
    if st.button("↩️ データを1つ前の状態に戻す (Undo)", use_container_width=True):
        if st.session_state.prev_working_df is not None:
            # 過去の箱から現在の箱へデータを戻す！
            st.session_state.working_df = st.session_state.prev_working_df.copy()
            st.success("1つ前のデータに戻りました。")
            st.rerun() # 画面をリロードして反映
        else:
            st.warning("戻せる過去のデータがありません。")

    # 📥 セーブ機能（現在のデータをpklファイルとしてダウンロード）
    if st.session_state.working_df is not None:
        # 保存したいデータを辞書にまとめる
        project_data = {
            "working_df": st.session_state.working_df,
            "chat_history": st.session_state.chat_history
        }
        # pickleを使ってPythonのデータをそのままバイナリ（0と1の塊）に変換
        pickled_data = pickle.dumps(project_data)
        
        st.download_button(
            label="💾 現在のプロジェクトを保存 (.pkl)",
            data=pickled_data,
            file_name="lab_project.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )

    # 📤 ロード機能（保存したpklファイルをアップロードして復元）
    uploaded_project = st.file_uploader("📂 保存したプロジェクトを読み込む", type=["pkl"])
    if uploaded_project is not None:
        if st.button("🚀 プロジェクトを復元する", type="primary"):
            try:
                loaded_data = pickle.loads(uploaded_project.read())
                st.session_state.working_df = loaded_data["working_df"]
                st.session_state.chat_history = loaded_data.get("chat_history", [])
                st.session_state.prev_working_df = None # 復元直後は過去の履歴をリセット
                # 👇 ✨ 追加：復元したデータも「初期状態」として記憶させる
                st.session_state.original_working_df = loaded_data["working_df"].copy()
                st.success("プロジェクトを復元しました！")
                st.rerun()
            except Exception as e:
                st.error(f"復元エラー（ファイルが壊れている可能性があります）: {e}")
    # 👆 ✨ 追加 ▲▲▲

# ==========================================
# 🔄 ファイル読み込み関数（CSV / Excel対応）
# ==========================================
def load_file(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
        #⚠️ pandasのread_csv関数でCSVファイルをデータフレームに変換して返す
    elif file.name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    return None

# ==========================================
# 🖥️ 4. メイン画面（3つのタブで分割）
# ==========================================
tab1, tab2, tab3 = st.tabs(["📁 1. データ準備＆マージ", "📊 2. グラフ＆レポート", "💬 3. AI計算アシスタント"])

# ------------------------------------------
# 📁 タブ1：データの読み込み・結合・クレンジング
# ------------------------------------------
with tab1:
    st.header("1. 実験データのアップロード")
    col1, col2 = st.columns(2)
    with col1:
        # ✨ 機能追加：xlsxファイルの受け取り許可！
        # 👇 ✨ 変更：複数ファイルの同時アップロードを許可（accept_multiple_files=True）
        file_a_list = st.file_uploader("📄 データA（メインデータ） ※複数選択可", type=["csv", "xlsx"], key="file_a", accept_multiple_files=True)
        #⚠️st.file_uploader でファイルをもらう
        
        if file_a_list:
            if len(file_a_list) == 1:
                # ファイルが1つだけなら今まで通り読み込む
                st.session_state.raw_df_a = load_file(file_a_list[0])
                #⚠️ もらったファイルを load_file 関数に渡してデータフレーム化し、セッションステートに保存する
            else:
                # 👇 ✨ 追加：複数ファイルがアップロードされた場合の「アンサンブル（自動平均化）」機能！
                st.info(f"📁 {len(file_a_list)} 個のファイルがアップロードされました。")
                ensemble_mode = st.radio("複数ファイルの処理方法", ["単純に縦に繋ぐ (Concat)", "基準列を揃えて平均・誤差を計算 (アンサンブル)"])
                
                if ensemble_mode == "単純に縦に繋ぐ (Concat)":
                    dfs = [load_file(f) for f in file_a_list]
                    st.session_state.raw_df_a = pd.concat(dfs, ignore_index=True)
                else:
                    # 最初のファイルを基準に、揃えるための列を選ばせる
                    temp_df = load_file(file_a_list[0])
                    base_col = st.selectbox("🔑 基準となる列（時間や電圧など、X軸になる列）", temp_df.columns.tolist())
                    
                    if st.button("🚀 アンサンブル計算を実行", type="primary"):
                        with st.spinner("全ファイルの平均と標準偏差を計算中..."):
                            try:
                                # プロの魔法：全ファイルを読み込み、基準列をインデックスにしてリスト化
                                dfs = [load_file(f).set_index(base_col) for f in file_a_list]
                                
                                # それらを3次元的に重ね合わせる
                                concatenated = pd.concat(dfs, keys=range(len(dfs)))
                                
                                # 元のインデックス（基準列）ごとに、平均と標準偏差を一気に計算！
                                mean_df = concatenated.groupby(level=1).mean()
                                std_df = concatenated.groupby(level=1).std().add_suffix('_std') # 標準偏差の列名には _std をつける
                                
                                # 平均と標準偏差を横に結合し、基準列をインデックスから普通の列に戻す
                                final_df = pd.concat([mean_df, std_df], axis=1).reset_index()
                                
                                st.session_state.raw_df_a = final_df
                                st.success("✅ アンサンブル計算が完了しました！下へ進んでください。")
                            except Exception as e:
                                st.error(f"計算エラー（各ファイルのデータの形が合っていない可能性があります）: {e}")

    with col2:
        file_b = st.file_uploader("📄 データB（結合用・任意）", type=["csv", "xlsx"], key="file_b")
        if file_b: st.session_state.raw_df_b = load_file(file_b)
            
    st.divider()

    # ▼▼▼ ✨機能追加：Vision OCR (画像からデータ抽出) ▼▼▼
    if use_ai:
        st.subheader("🖼️ 方法2：画像（スクショ等）からデータを抽出")
        st.write("実験データの表が写っている画像をアップロードしてください。AIが表を読み取り、データフレームに変換します。")
        
        file_img = st.file_uploader("表の画像をアップロード", type=["png", "jpg", "jpeg"], key="file_img")
        
        if file_img:
            st.image(file_img, caption="アップロードされた画像", width=300)
            
            if st.button("🔄 画像からデータを抽出する", type="primary"):
                if not use_ai:
                    st.error("AIアシスト機能がオフになっています。左のサイドバーからオンにしてください。")
                else:
                    with st.spinner("AIが画像を解析して表を抽出しています...（少し時間がかかります）"):
                        try:
                            # 1. 脳みそ（Gemini）の準備（Vision能力を持つモデルを指定）
                            llm = ChatGoogleGenerativeAI(model=ai_model)

                            # 2. 画像をBase64テキストに変換
                            image_data = base64.b64encode(file_img.read()).decode('utf-8')
                            
                            # 3. AIへのメッセージ（テキスト＋画像）を作成
                            message = HumanMessage(
                                content=[
                                    {
                                        "type": "text", 
                                        "text": """
                                        あなたは優秀なデータアナリストです。
                                        アップロードされた画像に写っている表を読み取り、そのデータを「CSV形式のテキスト」として出力してください。
                                        
                                        【出力の制約】
                                        1. 最初の行をヘッダー（列名）にしてください。
                                        2. 列名やデータの中に「,（カンマ）」が含まれる場合は、その値を「"（ダブルクォーテーション）」で囲んでください。
                                        3. CSVテキスト以外の説明文（「はい、承知しました」など）は一切出力しないでください。CSVデータのみを出力してください。
                                        4. データのケタ数は、画像に写っている通りに保持してください。
                                        """
                                    },
                                    {
                                        "type": "image_url", 
                                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                                    }
                                ]
                            )
                            
                            # 4. Geminiを呼び出して、CSVテキストをもらう
                            response = llm.invoke([message])
                            csv_text = response.content
                            
                            # 5. もらったCSVテキストを、Pandasのデータフレームに変換
                            # StringIO は、ただの文字列を「ファイル」のように扱わせてくれる魔法の道具です
                            try:
                                ocr_df = pd.read_csv(StringIO(csv_text))
                                
                                # 6. 成功！データAとして記憶させる
                                st.session_state.raw_df_a = ocr_df
                                st.success("✅ 画像からのデータ抽出に成功しました！下の表で確認してください。")
                            except Exception as e_pandas:
                                st.error(f"データの変換に失敗しました（AIが出力したCSVの形式が正しくありません）。もう一度試すか、別の画像を試してください。\nAIの出力: {csv_text}\nエラー: {e_pandas}")

                        except Exception as e_ai:
                            st.error(f"AIの呼び出しでエラーが発生しました: {e_ai}")

        st.divider()
    # ▲▲▲ ここまで機能追加 ▲▲▲

    st.header("2. データの結合とクレンジング（外れ値除外）")
    
    if st.session_state.raw_df_a is not None:
        if st.session_state.raw_df_b is not None:
            merge_col = st.text_input("🔑 結合の基準にする列名（空白は横連結）")
            if st.button("🔄 データを結合"):
                progress_text = "データを結合しています..."
                my_bar = st.progress(0, text=progress_text)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    my_bar.progress(percent_complete + 1, text=progress_text)
                try:
                    if merge_col: st.session_state.working_df = pd.merge(st.session_state.raw_df_a, st.session_state.raw_df_b, on=merge_col)
                    else: st.session_state.working_df = pd.concat([st.session_state.raw_df_a, st.session_state.raw_df_b], axis=1)
                    my_bar.empty()
                    #.empty() は用が済んだUIパーツを画面から跡形もなく消滅させるという機能

                    # 👇 ✨ 追加：結合が成功したら、それを初期状態として保存する
                    st.session_state.original_working_df = st.session_state.working_df.copy()

                except Exception as e: st.error(f"結合エラー: {e}")
        elif st.session_state.working_df is None:
            st.session_state.working_df = st.session_state.raw_df_a.copy()
            # 👇 ✨ 追加：データBがない場合も初期状態として保存
            st.session_state.original_working_df = st.session_state.raw_df_a.copy()

        if st.session_state.working_df is not None:
            st.info("💡 左端のチェックボックスを選択し、Deleteキーで行ごと削除（外れ値除外）できます。")

            # 👇 ✨ 追加：クレンジングをリセットして初期状態に戻すボタン
            if st.button("🔄 削除・編集を取り消して初期状態に戻す"):
                if st.session_state.original_working_df is not None:
                    st.session_state.working_df = st.session_state.original_working_df.copy()
                    
                    # ✨ 最強の魔法：鍵の数字を+1する！
                    # （鍵が変わると、Streamlitは「新しいエディタだ！」と勘違いして、過去の編集履歴を完全に忘れます）
                    st.session_state.editor_key += 1 
                    
                    st.success("初期状態にリセットしました！")
                    st.rerun() # 画面をリロードしてエディタを元に戻す

            # 👇 ✨ ここから追加：列（カラム）の選択機能
            st.markdown("**🗑️ 列（カラム）のクレンジング**")
            all_cols = st.session_state.working_df.columns.tolist()
            # マルチセレクトで残したい列を選ばせる（デフォルトは最初全部入った状態）
            selected_cols = st.multiselect(
                "📌 分析に残したい列を選択してください（不要な列は × ボタンで消せます）", 
                options=all_cols, 
                default=all_cols
            )
            
            if not selected_cols:
                st.warning("⚠️ 少なくとも1つの列を残してください。")
            else:
                # ユーザーが選んだ列だけに絞り込んだ表を作る
                df_to_edit = st.session_state.working_df[selected_cols]

                # 👇 ✨ 変更：一番最後に 名前：key=... を追加！
                edited_df = st.data_editor(
                    df_to_edit.round(sig_figs),
                    num_rows="dynamic", 
                    use_container_width=True,
                    key=f"data_editor_{st.session_state.editor_key}" # 👈 ここで鍵を使う！
                )
                #⚠️ st.data_editor でデータフレームを綺麗にし、編集後のデータを edited_df に保存する
                #⚠️綺麗になったデータを px.Scatter (Plotly) や LangChain (AI) に渡す
                if st.button("💾 このデータでグラフを作成する (確定)", type="primary"):
                    # 👇 ✨ 追加：上書きされる前に、現在のデータを過去の箱にコピーしておく！
                    st.session_state.prev_working_df = st.session_state.working_df.copy()

                    st.session_state.working_df = edited_df
                    st.success("✅ データを確定しました！「📊 2. グラフ＆レポート」へ進んでください。")
    else:
        st.warning("⚠️ 方法1でファイルをアップロードするか、方法2で画像からデータを抽出してください。")

# ------------------------------------------
# 📊 タブ2：最強のグラフ描画エンジン ＋ レポート出力
# ------------------------------------------
with tab2:
    st.header("📊 2. グラフとレポートの出力")
    
    if st.session_state.working_df is not None:
        df = st.session_state.working_df
        cols = df.columns.tolist()

        with st.expander("🛠️ グラフの設定を開く", expanded=True):
        #with は「ここからここまでが一つの箱だよ」と宣言する機能です。
        #st.expander（折りたたみ箱）の中に st.columns(3)（3分割の箱）を作り、
        #さらにその c1（左側の箱）に部品を入れる…というマトリョーシカ構造
            # ✨ 機能追加3：Z軸（奥行き）の選択肢を追加！
            c1, c2, c3, c4 = st.columns(4)
            with c1: x_col = st.selectbox("➡️ 横軸 (X)", cols)
            with c2: y_cols = st.multiselect("⬆️ 縦軸 (Y) ※2D用は複数可", cols, default=[cols[1]] if len(cols)>1 else None)
            with c3: z_col = st.selectbox("↕️ 奥行き (Z) ※3D・等高線用", ["なし"] + cols)
            with c4: graph_type = st.selectbox("📈 グラフの種類", ["折れ線グラフ", "散布図", "棒グラフ", "3D散布図", "3D曲面", "等高線マップ"])

            st.divider()
            # ✨ 機能追加：波形解析メニュー！
            st.markdown("**🔬 波形解析設定 (物理・電気・振動データ向け)**")
            wave1, wave2, wave3 = st.columns(3)
            with wave1:
                use_smooth = st.checkbox("〰️ スムージング (ノイズ除去)")
                if use_smooth:
                    smooth_window = st.slider("平滑化の強さ (奇数)", 5, 51, 11, step=2)
            with wave2:
                use_peaks = st.checkbox("📍 ピーク（山の頂点）自動検出")
                if use_peaks:
                    peak_prom = st.number_input("ピークの目立ちやすさ", min_value=0.0, value=0.1, step=0.1)
            with wave3:
                use_fft = st.checkbox("🌊 FFT (周波数スペクトル解析)")
                if use_fft:
                    dt = st.number_input("サンプリング間隔 (X軸の1目盛りの秒数)", value=0.001, format="%.6f")

            st.divider()
            # ✨ 機能追加：タイトルと軸ラベルの自由設定！
            st.markdown("**📝 ラベル設定**")
            lbl1, lbl2, lbl3 = st.columns(3)
            with lbl1: graph_title = st.text_input("グラフのタイトル", value="実験データ解析グラフ")
            with lbl2: x_title = st.text_input("横軸 (X) のタイトル", value=x_col)
            with lbl3: y_title = st.text_input("縦軸 (Y) のタイトル", value="Y軸の値")

            st.divider()
            st.markdown("**👑 実験解析・プロフェッショナル設定**")
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                is_log_x = st.checkbox("📐 X軸を対数スケールに")
                is_log_y = st.checkbox("📐 Y軸を対数スケールに")
                use_subplot = st.checkbox("🪟 2つのグラフを上下に分割") if len(y_cols) >= 2 else False
                # 👇 ✨ 追加：グリッド線のチェックボックス（デフォルトはON）
                show_grid = st.checkbox("▦ グラフにグリッド線を表示", value=True)
            with adv2:
                err_col = st.selectbox("📏 誤差棒（エラーバー）の列", ["なし"] + cols)
                use_trend = st.checkbox("📈 近似曲線と数式を表示")
                trend_type = st.selectbox("曲線の種類", ["1次式 (線形)", "2次式", "3次式"]) if use_trend else None
            with adv3:
                use_theory = st.checkbox("✍️ 理論値カーブを重ね描き")
                theory_eq = st.text_input("理論式 (例: 9.8 * x**2 / 2)", value="x**2") if use_theory else ""

        # 👇 ✨ 変更：ボタンを押したら「スイッチをON（True）」にするだけ！
        if st.button("✨ グラフを描画する！", type="primary", use_container_width=True):
            st.session_state.show_graph = True 

        # 👇 ✨ 追加：スイッチがON（True）なら、常にグラフを描画し続ける！
        if st.session_state.show_graph:
            with st.spinner("グラフを生成中..."):
                # ▼▼▼ 🌌 3D・等高線マップの描画モード ▼▼▼
                if graph_type in ["3D散布図", "3D曲面", "等高線マップ"]:
                    if z_col == "なし":
                        st.error("⚠️ 3Dや等高線を描画するには「↕️ 奥行き (Z)」の列を選択してください！")
                    elif not y_cols or len(y_cols) > 1:
                        st.error("⚠️ 3Dや等高線を描画する場合、「⬆️ 縦軸 (Y)」は1つだけ選んでください！")
                    else:
                        y_col = y_cols[0]
                        
                        # X, Y, Zのデータを数値化して、エラーがない行だけを取り出す
                        x_val = pd.to_numeric(df[x_col], errors='coerce')
                        y_val = pd.to_numeric(df[y_col], errors='coerce')
                        z_val = pd.to_numeric(df[z_col], errors='coerce')
                        valid_idx = x_val.notna() & y_val.notna() & z_val.notna()
                        x_val, y_val, z_val = x_val[valid_idx], y_val[valid_idx], z_val[valid_idx]
                        
                        fig = go.Figure()

                        # 🟢 3D散布図（空間に点を打つ）
                        if graph_type == "3D散布図":
                            fig.add_trace(go.Scatter3d(
                                x=x_val, y=y_val, z=z_val, mode='markers',
                                marker=dict(size=5, color=z_val, colorscale='Viridis', showscale=True)
                            ))
                            fig.update_layout(scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))

                        # 🔵 3D曲面 ＆ 等高線マップ（空間を面で繋ぐ魔法）
                        else:
                            # 1. バラバラの点を、綺麗な網目（メッシュ）に変換する
                            xi = np.linspace(x_val.min(), x_val.max(), 50)
                            yi = np.linspace(y_val.min(), y_val.max(), 50)
                            X, Y = np.meshgrid(xi, yi)
                            # 2. SciPyのgriddataを使って、XとYの交点のZの高さを推測（補間）する
                            Z = griddata((x_val, y_val), z_val, (X, Y), method='linear')
                            
                            if graph_type == "3D曲面":
                                fig.add_trace(go.Surface(x=xi, y=yi, z=Z, colorscale='Plasma'))
                                fig.update_layout(scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col))
                            elif graph_type == "等高線マップ":
                                fig.add_trace(go.Contour(x=xi, y=yi, z=Z, colorscale='Plasma', contours=dict(showlabels=True)))
                                fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)

                        fig.update_layout(height=700, title_text=f"{graph_type} ({z_col})", template=theme)
                        st.plotly_chart(fig, use_container_width=True)
                
                # ▼▼▼ 📊 2Dグラフの描画モード ▼▼▼
                else:
                    if not y_cols:
                        st.warning("⚠️ Y軸を1つ以上選択してください。")
                    else:
                        #⚠️1. キャンバス（画用紙）の準備
                        #rows=2 なら上下2画面、rows=1 なら1画面の画用紙を用意する
                        rows = 2 if use_subplot else 1
                        fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                        x_data = pd.to_numeric(df[x_col], errors='coerce').dropna()
                        #グラフを描くとき、データの中に1つでも「文字」が混ざっているとエラーになる
                        #pd.to_numeric（数値） は「数字っぽいやつをちゃんとした数値に変換する」機能
                        #errors='coerce'は変換できないやつは エラーで止まらずに『NaN（欠損値）になる
                        #dropna()はNaNを消す機能。
                        #つまり「数字じゃないやつを消して、ちゃんとした数値だけでグラフを描いてね」という意味になります。

                        fft_results = [] # FFTの結果を保存する箱

                        for i, y_col in enumerate(y_cols):
                            y_data = pd.to_numeric(df[y_col], errors='coerce')
                            common_idx = x_data.index.intersection(y_data.dropna().index)
                            x_val, y_val = x_data[common_idx], y_data[common_idx]
                            row_idx = 2 if (use_subplot and i >= 1) else 1

                            # 〰️ スムージング処理
                            if use_smooth and len(y_val) > smooth_window:
                                # Savitzky-Golayフィルタというプロ御用達のノイズ除去魔法
                                y_val = pd.Series(savgol_filter(y_val, smooth_window, 3), index=y_val.index)
                            
                            error_y_dict = None
                            if err_col != "なし":
                                e_data = pd.to_numeric(df[err_col], errors='coerce')[common_idx]
                                error_y_dict = dict(type='data', array=e_data, visible=True)

                            #⚠️2. データの流し込み（★ここが一番重要！）📈 メインデータの描画
                            #fig.add_trace() というペンを使って、選ばれたY軸の数だけ繰り返し（forループ）で
                            #グラフを描き足していきます
                            mode = 'markers' if graph_type == "散布図" else ('lines+markers' if graph_type == "折れ線グラフ" else 'lines')
                            # 👇 ✨ 変更：一番最後に customdata=common_idx を追加！
                            #投げ縄のためにfig.add_trace(...) の中に、customdata=common_idx という「元の行番号を隠し持たせる追跡マーカー（customdata）を仕込む」
                            if graph_type == "棒グラフ": 
                                fig.add_trace(go.Bar(x=x_val, y=y_val, name=y_col, error_y=error_y_dict, customdata=common_idx.tolist()), row=row_idx, col=1)
                            else: 
                                fig.add_trace(go.Scatter(x=x_val, y=y_val, mode=mode, name=y_col, error_y=error_y_dict, customdata=common_idx.tolist()), row=row_idx, col=1)
                            # 📍 ピーク検出処理
                            if use_peaks:
                                # numpy配列に変換してから処理
                                y_np = y_val.to_numpy()
                                peaks, _ = find_peaks(y_np, prominence=peak_prom)
                                if len(peaks) > 0:
                                    fig.add_trace(go.Scatter(
                                        x=x_val.iloc[peaks], y=y_val.iloc[peaks], 
                                        mode='markers', marker=dict(color='red', symbol='x', size=10),
                                        name=f'{y_col} ピーク'
                                    ), row=row_idx, col=1)

                            # 🌊 FFT計算処理（後で別のグラフとして描くために保存）
                            if use_fft:
                                N = len(y_val)
                                yf = np.fft.fft(y_val.to_numpy())
                                xf = np.fft.fftfreq(N, d=dt)
                                mask = xf > 0 # プラスの周波数だけ取り出す
                                fft_results.append((y_col, xf[mask], np.abs(yf[mask]) / (N/2)))

                            if use_trend and len(x_val) > 1:
                                deg = 1 if "1" in trend_type else (2 if "2" in trend_type else 3)
                                z = np.polyfit(x_val, y_val, deg)
                                p = np.poly1d(z)
                                x_trend = np.linspace(x_val.min(), x_val.max(), 100)
                                fig.add_trace(go.Scatter(x=x_trend, y=p(x_trend), mode='lines', line=dict(dash='dash', width=2), name=f'{y_col} 近似'), row=row_idx, col=1)
                                eq_str = f"y = " + " + ".join([f"{c:.2e}x^{deg-j}" if (deg-j)>0 else f"{c:.2e}" for j, c in enumerate(z)])
                                #近似曲線の数式（$y = ax^2 + bx + c$ など）の文字列を組み立てている部分
                                #「リスト内包表記」と言い、Pythonでは [ 処理 for 変数 in リスト ] という形で書く
                                #{c:.2e} は「cを有効数字2桁の指数表記で表示してね」という意味
                                #この1行だけで、「係数を綺麗な文字にして、xの何乗かをつけて、全部 + で繋ぐ」という処理をする
                                fig.add_annotation(text=eq_str, x=x_val.mean(), y=y_val.max(), showarrow=False, font=dict(color="red"), row=row_idx, col=1)

                        if use_theory and theory_eq:
                            try:
                                x_range = np.linspace(x_data.min(), x_data.max(), 200)
                                y_theory = eval(theory_eq, {"x": x_range, "np": np})
                                #eval()はただの文字列を、Pythonのプログラムコードとして無理やり実行する関数
                                #データを全消去するコードを打たれるなどの危険もあるので、
                                #{"x": x_range, "np": np} と付けて、
                                # あなたが使えるのは変数xと、NumPy（np）だけですよと制限している
                                fig.add_trace(go.Scatter(x=x_range, y=y_theory, mode='lines', line=dict(color='orange', width=3), name='理論値'), row=1, col=1)
                            except Exception as e: st.error(f"⚠️ 理論式の計算エラー: {e}")
                            #⚠️エラーを絶対に止めない try-except
                            #ユーザーは必ず想定外の操作（変なファイルを上げる、数式を間違えるなど）をします
                            #とりあえず try: の中で危ない処理（結合やAIの呼び出しなど）をさせて、
                            #もし爆発したら except: でエラーメッセージだけを画面に出して
                            #プログラム全体は止めないという安全装置の概念

                        #⚠️3. レイアウトの調整（タイトル、軸ラベル、テーマ、対数スケールなど）
                        fig.update_layout(
                            height=600 if rows==2 else 450, 
                            title_text=graph_title, 
                            xaxis_title=x_title, 
                            yaxis_title=y_title,
                            template=theme
                        )
                        if is_log_x: fig.update_xaxes(type="log")
                        if is_log_y: fig.update_yaxes(type="log")

                        # 👇 ✨ ここを追加：グリッド線のON/OFF制御
                        if show_grid:
                            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
                        else:
                            fig.update_xaxes(showgrid=False)
                            fig.update_yaxes(showgrid=False)

                        #⚠️4. グラフの表示
                        # 👇 ✨ 変更：グラフを表示しつつ、ユーザーの「選択（投げ縄）」を受け取る！
                        # 👇 ✨ 変更：key=f"my_main_chart_{...}" を追加してグラフに名前をつける！
                        selection_event = st.plotly_chart(
                            fig, 
                            key=f"my_main_chart_{st.session_state.editor_key}", # 👈 これがないと投げ縄を忘れてしまいます！
                            use_container_width=True, 
                            on_select="rerun", # 選択されたら即座に再計算する
                            selection_mode=["lasso", "box"] # 投げ縄と四角形選択を許可
                        )
                        # 👇 ✨ 追加：もし投げ縄で選択されたデータ（点）があれば、削除UIを出現させる
                        if selection_event and len(selection_event.selection.get("points", [])) > 0:
                            selected_points = selection_event.selection["points"]
                            
                            # 仕込んだ customdata（元の行番号）を取り出す
                            target_indices = []
                            for pt in selected_points:
                                if "customdata" in pt:
                                    # Plotlyの仕様でリストに入っている場合があるので取り出す
                                    idx = pt["customdata"][0] if isinstance(pt["customdata"], list) else pt["customdata"]
                                    target_indices.append(idx)
                            
                            target_indices = list(set(target_indices)) # 重複を消す
                            
                            st.error(f"🎯 グラフ上で {len(target_indices)} 個のデータ（外れ値）が選択されています！")
                            
                            if st.button("🗑️ 選択した外れ値をデータから完全に削除して再描画", type="primary"):
                                # タイムマシン用にバックアップ
                                st.session_state.prev_working_df = st.session_state.working_df.copy()
                                
                                # Pandasのdrop機能で、該当の行番号を一気に消し去る！
                                st.session_state.working_df = st.session_state.working_df.drop(index=target_indices)
                                
                                # エディタの記憶もリセット（前回学んだ動的キーの魔法！）
                                st.session_state.editor_key += 1
                                
                                st.success("外れ値を削除しました！")
                                st.rerun() # 画面をリロードして反映

                        # 🌊 FFTグラフの描画（選択された場合のみ、メイングラフの下に出現！）
                        if use_fft and fft_results:
                            st.subheader("🌊 FFT (周波数スペクトル) 解析結果")
                            fig_fft = go.Figure()
                            for name, xf, yf in fft_results:
                                fig_fft.add_trace(go.Scatter(x=xf, y=yf, mode='lines', name=name))
                            fig_fft.update_layout(height=400, xaxis_title="周波数 (Hz)", yaxis_title="振幅", template=theme)
                            st.plotly_chart(fig_fft, use_container_width=True)

                        # ✨ 機能追加3：裏側で使ったPythonコードのエクスポート！
                        st.subheader("💻 Python 描画コード (Plotly)")
                        st.write("このグラフをローカル環境で再現するためのPythonコードです。")
                        python_code = f"""import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# データの読み込み (ご自身のファイルパスに変更してください)
# df = pd.read_csv("your_data.csv")

# グラフの設定
fig = make_subplots(rows={rows}, cols=1, shared_xaxes=True)

# 描画処理 (代表的な列の例)
fig.add_trace(
    go.Scatter(x=df['{x_col}'], y=df['{y_cols[0]}'], mode='{mode}', name='{y_cols[0]}'),
    row=1, col=1
)

# レイアウトの更新
fig.update_layout(
    title_text="{graph_title}",
    xaxis_title="{x_title}",
    yaxis_title="{y_title}",
    template="{theme}"
)
fig.show()
"""
                        st.code(python_code, language="python")

        st.divider()
        st.subheader("📝 レポート用 表出力（LaTeX / Markdown）")
        c_out1, c_out2 = st.columns(2)
        rounded_df = df.round(sig_figs)
        with c_out1:
            st.markdown("**▼ Markdown形式 (Word等へのコピペ用)**")
            st.code(rounded_df.to_markdown(), language="markdown")
        with c_out2:
            st.markdown("**▼ LaTeX形式**")
            try:
                latex_code = rounded_df.style.to_latex()
            except:
                latex_code = rounded_df.to_latex()
            st.code(latex_code, language="latex")
        
        # 👇 ✨ 追加：表を画像として保存する機能
        st.divider()
        st.subheader("📸 表を画像として保存 (PNG)")
        st.write("右上のカメラアイコン(📷)を押すと、下の表をそのまま画像としてダウンロードできます。")
        
        # Plotlyの「Table」機能を使って表を描画する
        table_fig = go.Figure(data=[go.Table(
            header=dict(values=list(rounded_df.columns), fill_color='paleturquoise', align='center', font=dict(size=14, color='black')),
            cells=dict(values=[rounded_df[col] for col in rounded_df.columns], fill_color='lavender', align='center', font=dict(size=12, color='black'))
        )])
        # 表の余白をなくしてスッキリさせる
        table_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=400)
        st.plotly_chart(table_fig, use_container_width=True)
        # 👆 ✨ 追加ここまで

        st.divider()
        
        # ✨ AIアシスト機能のON/OFF制御
        if use_ai:
            st.subheader("🤖 AIによる実験データの自動考察")
            if st.button("📝 現在のデータから実験レポートの考察案を作成する"):
                with st.spinner("AIがデータの統計情報と傾向を分析しています..."):
                    try:
                        llm = ChatGoogleGenerativeAI(model=ai_model)
                        data_summary = df.describe().to_markdown()
                        #AIの「脳の容量（トークン）」を節約する工夫
                        #膨大なデータに対してAIはエラーを起こしてしまう
                        #そこで、Pandasの describe() という関数を使い,
                        #データの平均や標準偏差などの統計情報だけを抜き取って、AIに渡すことで
                        #少ない文字数でデータの全体像を正確に把握させ、爆速で考察を作らせている
                        #プロンプトエンジニアリング
                        prompt = f"""
                        あなたは優秀な理系の大学教授です。以下の実験データの統計情報を見て、学生のレポート用の「考察のヒント」を作成してください。
                        【実験データの統計情報】\n{data_summary}\n
                        【出力形式】\n1. データの全体的な傾向\n2. 標準偏差（バラつき）から言えること\n3. 理論値との比較をする際のアドバイス
                        """
                        response = llm.invoke(prompt)
                        st.success("✨ 考察の生成が完了しました！")
                        st.markdown(response.content)
                    except Exception as e: st.error(f"AIの呼び出しエラー: {e}")
        else:
            st.info("💡 AIによる自動考察を利用する場合は、左のサイドバーから「AIアシスト機能」をオンにしてください。")

    else:
        st.warning("⚠️ まずは「📁 1. データ準備」タブでデータを確定させてください。")

# ------------------------------------------
# 💬 タブ3：AI計算アシスタント（Pandas Agent）
# ------------------------------------------
with tab3:
    st.header("💬 3. AIデータ計算アシスタント")
    
    if not use_ai:
        st.warning("⚠️ AIデータ計算アシスタントを利用するには、左のサイドバーから「AIアシスト機能を有効にする」をオンにしてください。")
    elif st.session_state.working_df is not None:
        st.markdown("AIが直接データ（Pandas）を操作します。「〇〇列と〇〇列を足して新しい列を作って」などと指示してください。")
        with st.expander("現在のデータを確認", expanded=True):
            st.dataframe(st.session_state.working_df.head().style.format(precision=sig_figs))

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("例：列Aと列Bを掛けて、新しい列を作って"):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            with st.chat_message("assistant"):
                with st.spinner("AIがデータを計算中...🧮"):
                    try:
                        llm = ChatGoogleGenerativeAI(model=ai_model, temperature=0)
                        agent = create_pandas_dataframe_agent(llm, st.session_state.working_df, verbose=True, allow_dangerous_code=True)
                        #「AI計算アシスタント（Pandas Agent）」を呼び出している部分
                        #AIは裏側で本物のPythonコードを自力でタイピングして、それをあなたのパソコン（サーバー）上で実際に実行している
                        #allow_dangerous_code=True は、「AIにPythonコードを書かせて、それを実際に動かす権限を与えるよ！」という許可証
                        
                        # 👇 ✨ 追加：AIがデータを破壊（変更）してしまう前にバックアップ！
                        st.session_state.prev_working_df = st.session_state.working_df.copy()

                        result = agent.invoke(prompt)
                        answer = result["output"]
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        st.rerun() 
                    except Exception as e:
                        st.error(f"計算中にエラーが発生しました: {e}")
    else:
        st.warning("⚠️ まずは「📁 1. データ準備」タブでデータを確定させてください。")