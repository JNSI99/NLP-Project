"""GlobalPulse: inteligencia narrativa. No modifica los CSV ni realiza llamadas API.
Instalar: python -m pip install streamlit pandas numpy plotly
Ejecutar: python -m streamlit run dashboard_intelligence.py
Seleccione la carpeta de CSV y cluster_analysis en la barra lateral.
"""
from pathlib import Path
import io
import ast
import json
import unicodedata
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = r"C:\Users\Nicoe\OneDrive\Desktop\NLP Project"
SCHEMAS = {
    "gold_titles.csv": ["article_id", "title", "language", "title_en"],
    "gold_texts.csv": ["article_id", "title", "language", "body", "body_en", "title_en"],
    "texts.csv": ["article_id", "title", "url", "source", "language", "body", "extraction_status", "extraction_method"],
    "test_titles.csv": ["article_id", "title", "language", "title_en", "frame_analysis", "tone"],
    "final_test.csv": ["article_id", "title", "language", "title_en", "actor", "actor_role"],
    "embbedings_test.csv": ["article_id", "fragment_id", "fragment", "embedding"],
    "clusters_test.csv": ["article_id", "fragment_id", "fragment", "embedding", "cluster", "cluster_probability"],
}


def clean_csv(raw):
    """Preserva identificadores como texto y no convierte 'NA' en valor ausente."""
    last_error = None
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False,
                             encoding=encoding)
            df.columns = df.columns.str.strip()
            return df.loc[:, ~df.columns.str.match(r"^Unnamed:")].apply(lambda c: c.str.strip())
        except UnicodeDecodeError as exc:
            last_error = exc
    raise last_error


def unique_records(df, key):
    """Excluye todas las filas con claves contradictorias; colapsa copias idénticas."""
    if df.empty:
        return df.copy(), 0, 0
    blank = int(df[key].eq("").sum())
    work = df.loc[df[key].ne("")].drop_duplicates()
    conflicts = work.loc[work.duplicated(key, keep=False), key].nunique()
    return work.loc[~work.duplicated(key, keep=False)].copy(), blank, conflicts


def pareto_actors(values, target=0.8):
    """Conserva el actor que cruza el umbral y todos los empatados con él."""
    counts = values[values.ne("")].value_counts()
    if counts.empty:
        return pd.DataFrame(columns=["actor", "registros", "porcentaje", "acumulado", "incluido"])
    result = counts.rename_axis("actor").reset_index(name="registros")
    result = result.sort_values(["registros", "actor"], ascending=[False, True]).reset_index(drop=True)
    result["porcentaje"] = result.registros / result.registros.sum() * 100
    result["acumulado"] = result.porcentaje.cumsum()
    crossing = np.searchsorted(result.acumulado.to_numpy(), target * 100 - 1e-9)
    cutoff = result.iloc[min(crossing, len(result) - 1)].registros
    result["incluido"] = result.registros.ge(cutoff)
    return result


def show_table(df, key):
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Descargar tabla CSV", df.to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"{key}.csv", mime="text/csv", key=f"download_{key}")


def bar_counts(values, label, key):
    s = values.loc[values.ne("")].value_counts()
    if s.empty:
        st.info(f"No hay valores disponibles para {label}.")
        return
    table = s.rename_axis(label).reset_index(name="registros")
    st.plotly_chart(px.bar(table, x="registros", y=label, orientation="h"),
                    width="stretch", key=key)


def matrix_chart(matrix, key, percent=False, unit="Artículos", selectable=True,
                 row_label=None, col_label=None, coverage_denominator=None):
    if matrix.empty:
        st.info("No hay registros suficientes para esta comparación.")
        return
    row_label = row_label or matrix.index.name or "fila"
    col_label = col_label or matrix.columns.name or "columna"
    color = unit
    view = matrix
    if selectable:
        options = ["Recuentos", f"% por {row_label} (fila)", f"% por {col_label} (columna)"]
        if coverage_denominator is not None:
            options.append("% de artículos del medio con fragmentos")
        mode = st.radio("Mostrar matriz como", options,
                        index=3 if coverage_denominator is not None else (1 if percent else 0),
                        horizontal=True, key=f"mode_{key}")
        if mode == options[1]:
            view = matrix.div(matrix.sum(axis=1).replace(0, np.nan), axis=0).fillna(0) * 100
            color = mode
            st.caption(f"Cada fila ({row_label}) suma 100 % cuando contiene datos: distribución de {col_label} dentro de cada {row_label}.")
        elif mode == options[2]:
            view = matrix.div(matrix.sum(axis=0).replace(0, np.nan), axis=1).fillna(0) * 100
            color = mode
            st.caption(f"Cada columna ({col_label}) suma 100 % cuando contiene datos: distribución de {row_label} dentro de cada {col_label}.")
        elif coverage_denominator is not None and mode == options[3]:
            view = matrix.div(coverage_denominator.reindex(matrix.index).replace(0, np.nan), axis=0).fillna(0) * 100
            color = "% de artículos"
            st.caption("Artículos del medio en el cluster / artículos del medio con fragmentos, incluido ruido. Los temas se solapan; las filas pueden superar el 100 %.")
        else:
            st.caption(f"Recuentos de coincidencias entre {row_label} y {col_label}. Las proporciones por fila/columna se calculan sobre estas coincidencias.")
        selected_rows = st.multiselect(
            f"Valores de {row_label} en la matriz (vacío = todos)",
            matrix.index.tolist(), key=f"rows_{key}")
        selected_cols = st.multiselect(
            f"Valores de {col_label} en la matriz (vacío = todos)",
            matrix.columns.tolist(), key=f"cols_{key}")
        view = view.loc[selected_rows or matrix.index.tolist(),
                        selected_cols or matrix.columns.tolist()]
        st.caption("Los selectores cambian los valores visibles en ambos ejes y en la tabla. Los porcentajes mantienen el denominador completo de los filtros globales; no se recalculan al ocultar categorías.")
    fig = px.imshow(view, aspect="auto", color_continuous_scale="Blues",
                    labels={"color": color})
    st.plotly_chart(fig, width="stretch", key=key)
    st.dataframe(view.round(2), width="stretch")


def parse_embedding(raw):
    """Acepta listas JSON/Python y la representación NumPy con espacios y saltos de línea."""
    try:
        if not isinstance(raw, str) or not raw.strip() or "..." in raw or "…" in raw:
            return None
        raw = raw.strip()
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            try:
                obj = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                if not (raw.startswith("[") and raw.endswith("]")):
                    return None
                tokens = re.split(r"[\s,]+", raw[1:-1].strip())
                number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
                if not tokens or not all(re.fullmatch(number, token) for token in tokens):
                    return None
                obj = [float(token) for token in tokens]
        v = np.asarray(obj, dtype=float)
        if v.ndim != 1 or not v.size or not np.isfinite(v).all():
            return None
        norm = np.linalg.norm(v)
        return v / norm if np.isfinite(norm) and norm > 0 else None
    except (ValueError, TypeError, SyntaxError, RecursionError, OverflowError):
        return None


def network_plot(nodes, edges, key, bipartite=False):
    """Nodos con claves separadas del texto: un actor y un rol homónimos no se fusionan."""
    if not nodes:
        st.info("No hay nodos disponibles en esta selección.")
        return
    positions = {}
    groups = [[n for n in nodes if n["group"] == g] for g in ["Actor", "Rol"]] if bipartite else [nodes]
    for group_index, group in enumerate(groups):
        for i, node in enumerate(group):
            if bipartite:
                positions[node["id"]] = (group_index * 2 - 1, 1 - 2 * i / max(1, len(group) - 1))
            else:
                angle = 2 * np.pi * i / len(group)
                positions[node["id"]] = (np.cos(angle), np.sin(angle))
    fig = go.Figure()
    max_weight = max((e["weight"] for e in edges), default=1)
    for e in edges:
        x0,y0 = positions[e["left"]]; x1,y1 = positions[e["right"]]
        fig.add_trace(go.Scatter(x=[x0,x1], y=[y0,y1], mode="lines",
            line=dict(width=1 + 4 * e["weight"] / max_weight, color="rgba(130,150,180,.5)"),
            text=[e["hover"],e["hover"]], hovertemplate="%{text}<extra></extra>", showlegend=False))
    largest = max(n["size"] for n in nodes)
    for group in sorted({n["group"] for n in nodes}):
        subset = [n for n in nodes if n["group"] == group]
        fig.add_trace(go.Scatter(x=[positions[n["id"]][0] for n in subset],
            y=[positions[n["id"]][1] for n in subset], mode="markers+text", name=group,
            text=[n["label"] for n in subset], textposition="top center",
            customdata=[n["hover"] for n in subset], hovertemplate="%{customdata}<extra></extra>",
            marker=dict(size=[12 + 30 * np.sqrt(n["size"] / max(1,largest)) for n in subset],
                        color="#70b7ff" if group != "Rol" else "#ffb86b", line=dict(width=1,color="white"))))
    fig.update_layout(height=750, xaxis=dict(visible=False), yaxis=dict(visible=False),
                      margin=dict(l=70,r=70,t=50,b=40), hovermode="closest")
    st.plotly_chart(fig, width="stretch", key=key)


def report_section_kind(title):
    normalized = "".join(c for c in unicodedata.normalize("NFKD", title.casefold())
                         if not unicodedata.combining(c))
    if "narrativ" in normalized or "narrative" in normalized:
        return "Narrativas"
    if "cita" in normalized or "quote" in normalized:
        return "Citas y atribuciones"
    if any(word in normalized for word in ["hallazgo", "finding", "argument", "interpretacion", "tension", "contrapuest"]):
        return "Hallazgos y argumentos"
    if "subtema" in normalized or "subtopic" in normalized:
        return "Subtemas"
    if "tema principal" in normalized or "main topic" in normalized:
        return "Tema principal"
    return "Otras secciones"


def split_report_sections(content):
    """Separa por encabezados; conserva texto y secciones desconocidas sin inventar resúmenes."""
    result, title, lines = [], "Texto sin encabezado", []
    def flush():
        body = "\n".join(lines).strip()
        if body:
            result.append(dict(titulo=title, categoria=report_section_kind(title), texto=body))
    for line in content.splitlines():
        heading = re.fullmatch(r"\s{0,3}#{1,6}\s+(.+?)(?:\s+#+)?\s*", line)
        bold_heading = re.fullmatch(r"\s*\*\*(.+?)\*\*\s*:?\s*", line)
        if heading or bold_heading:
            flush()
            title = (heading or bold_heading).group(1).strip()
            lines = []
        else:
            lines.append(line)
    flush()
    return result


def main():
    st.set_page_config(page_title="GlobalPulse | Media Intelligence", page_icon="🌐", layout="wide")
    st.title("GlobalPulse · Media Intelligence")
    st.caption("Encuadres de titulares, cobertura temática y exploración de narrativas del corpus analizado.")
    folder = Path(st.sidebar.text_input("Carpeta con los CSV", ROOT))
    reports_folder = Path(st.sidebar.text_input("Carpeta de análisis", str(folder / "cluster_analysis")))
    uploads = st.sidebar.file_uploader("O sube CSV, análisis .md y el informe .txt", type=["csv", "md", "txt"], accept_multiple_files=True)
    st.sidebar.caption("Los archivos subidos tienen prioridad sobre los de la carpeta. No se modifican los originales.")
    uploaded = {f.name: f.getvalue() for f in uploads or []}
    issues, tables, loaded_files = [], {}, set()
    for name, cols in SCHEMAS.items():
        try:
            raw = uploaded.get(name)
            if raw is None:
                raw = (folder / name).read_bytes()
            df = clean_csv(raw)
            missing = set(cols) - set(df.columns)
            if missing:
                issues.append(f"{name}: faltan columnas {', '.join(sorted(missing))}. Se desactiva este archivo.")
                df = pd.DataFrame(columns=cols)
            else:
                loaded_files.add(name)
            # Solo columnas declaradas por el usuario; embedding no se necesita para estas métricas.
            tables[name] = df.reindex(columns=cols, fill_value="")
        except FileNotFoundError:
            tables[name] = pd.DataFrame(columns=cols)
            issues.append(f"No disponible: {name}.")
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            tables[name] = pd.DataFrame(columns=cols)
            issues.append(f"No se pudo leer {name}: {exc}")
    texts, blank, conflicts = unique_records(tables["texts.csv"], "article_id")
    issues.append(f"texts.csv: {blank} filas sin article_id y {conflicts} IDs contradictorios excluidos.")
    # Los recuentos de titulares conservan todas las filas de test_titles.
    # Final se usa exclusivamente para actores y roles, sin sus etiquetas de encuadre/tono.
    def attach_source(df):
        out = df.merge(texts[["article_id", "source", "url"]], on="article_id",
                       how="left", validate="many_to_one")
        out[["source", "url"]] = out[["source", "url"]].fillna("")
        return out
    articles = attach_source(tables["test_titles.csv"])
    roles = attach_source(tables["final_test.csv"][["article_id", "title", "title_en", "language", "actor", "actor_role"]])
    issues.append(f"test_titles.csv: {len(articles)} filas conservadas; {int(articles.source.eq('').sum())} sin medio vinculado.")
    metadata = texts[["article_id", "source", "language", "title", "url"]].copy()

    raw_clusters = tables["clusters_test.csv"].drop(columns=["embedding"]).drop_duplicates()
    invalid_key = raw_clusters.article_id.eq("") | raw_clusters.fragment_id.eq("")
    clusters = raw_clusters.loc[~invalid_key].copy()
    collision = clusters.duplicated(["article_id", "fragment_id"], keep=False)
    issues.append(f"Fragmentos excluidos: {int(invalid_key.sum())} sin claves y {int(collision.sum())} filas con claves contradictorias.")
    clusters = clusters.loc[~collision].copy()
    parsed = pd.to_numeric(clusters.cluster, errors="coerce")
    valid = parsed.notna() & parsed.ge(-1) & parsed.mod(1).eq(0)
    issues.append(f"Clusters con etiqueta inválida excluidos: {int((~valid).sum())}.")
    clusters = clusters.loc[valid].copy()
    clusters["cluster"] = parsed.loc[valid].astype(int)
    clusters = clusters.merge(metadata, on="article_id", how="left", validate="many_to_one")
    for col in ["source", "language", "title", "url"]:
        clusters[col] = clusters[col].fillna("")
    source_options = sorted(set(pd.concat([metadata.source, articles.source, roles.source]).loc[lambda x: x.ne("")]))
    languages = sorted(set(pd.concat([metadata.language, articles.language, roles.language, tables["gold_texts.csv"].language, tables["gold_titles.csv"].language]).loc[lambda x: x.ne("")]))
    selected_sources = st.sidebar.multiselect("Medios (vacío = todos)", source_options)
    selected_languages = st.sidebar.multiselect("Idiomas originales (vacío = todos)", languages)
    def filtered(df):
        out = df.copy()
        if selected_sources:
            out = out.loc[out.source.isin(selected_sources)]
        if selected_languages:
            out = out.loc[out.language.isin(selected_languages)]
        return out
    articles, roles, metadata, clusters = map(filtered, [articles, roles, metadata, clusters])
    st.sidebar.caption("Los filtros de medio/idioma excluyen registros cuya procedencia no se pueda vincular. El 80 % se recalcula dentro de la selección.")
    a, b, c, d = st.columns(4)
    a.metric("Artículos disponibles", len(tables["gold_texts.csv"]) if "gold_texts.csv" in loaded_files else "—",
             help="Número de filas de gold_texts.csv, sin deduplicar ni cruzar con otros archivos.")
    b.metric("Titulares disponibles", len(tables["gold_titles.csv"]) if "gold_titles.csv" in loaded_files else "—",
             help="Número de filas de gold_titles.csv, sin deduplicar ni cruzar con otros archivos.")
    c.metric("Medios identificados", metadata.loc[metadata.source.ne(""), "source"].nunique())
    d.metric("Temas / clusters", clusters.loc[clusters.cluster.ge(0), "cluster"].nunique())
    st.caption("Artículos y titulares disponibles: totales de gold_texts.csv y gold_titles.csv, sin aplicar filtros. "
               "Los encuadres y tonos proceden de test_titles.csv; actores y roles de final_test.csv, ambos sobre titulares; "
               "medios, clusters y gráficas reflejan los filtros seleccionados.")
    tabs = st.tabs(["Encuadres y medios", "Actores y roles", "Cobertura temática", "Informes", "Alcance y límites", "Idiomas", "Red de clusters", "Red de actores y roles", "Inteligencia narrativa"])

    with tabs[0]:
        st.subheader("Encuadres y tonos de los titulares")
        st.caption(f"Titulares en test_titles.csv dentro de los filtros: {len(articles)}. "
                   f"Sin encuadre: {int(articles.frame_analysis.eq('').sum())}; sin tono: {int(articles.tone.eq('').sum())}.")
        left, right = st.columns(2)
        with left:
            bar_counts(articles.frame_analysis, "frame_analysis", "frames")
        with right:
            bar_counts(articles.tone, "tone", "tones")
        st.caption("Las etiquetas se agrupan literalmente, sin inferir categorías narrativas nuevas. Se muestran todas las categorías no vacías de test_titles.csv.")
        known = articles.loc[articles.source.ne("") & articles.frame_analysis.ne("")]
        st.write("**Encuadres por medio**")
        matrix_chart(pd.crosstab(known.source, known.frame_analysis), "sources_frames", True, unit="Titulares", row_label="medio", col_label="encuadre")
        st.caption("Se incluyen titulares con medio y frame_analysis no vacíos, dentro de los filtros. Consulte el recuento y los textos antes de comparar muestras pequeñas.")
        show_table(known.groupby("source").size().rename("titulares_con_encuadre").reset_index(), "muestras_por_medio")
        st.write("**Tono × encuadre**")
        paired = articles.loc[articles.frame_analysis.ne("") & articles.tone.ne("")]
        matrix_chart(pd.crosstab(paired.frame_analysis, paired.tone), "frame_tone", True, unit="Titulares", row_label="encuadre", col_label="tono")
        show_table(articles, "titulares_filtrados")

    with tabs[1]:
        st.subheader("Actores que acumulan al menos el 80 %")
        normalize = st.checkbox("Unificar US / U.S. / the United States / United States", value=True)
        actor_values = roles.actor.copy()
        if normalize:
            actor_values = actor_values.map(lambda x: "United States" if x.casefold() in {"us", "u.s.", "the united states", "united states"} else x)
        ranking = pareto_actors(actor_values)
        st.caption("Se cuenta un valor de actor por fila de final_test.csv. No se separan listas ni se corrigen actores ambiguos. Se incluyen los empatados en el punto de corte; puede superarse el 80 %.")
        if ranking.empty:
            st.info("No hay actores identificados disponibles.")
        else:
            chosen = ranking.loc[ranking.incluido]
            covered = chosen.porcentaje.sum()
            st.metric("Cobertura del conjunto mostrado", f"{covered:.1f} %", f"{len(chosen)} de {len(ranking)} actores")
            fig = px.bar(chosen, x="actor", y="registros", hover_data=["porcentaje", "acumulado"])
            st.plotly_chart(fig, width="stretch")
            st.warning("La frecuencia no valida las identificaciones. Los errores frecuentes también pueden aparecer aquí. El tono del titular no se interpreta como sentimiento hacia el actor.")
            show_table(ranking, "ranking_actores_completo")
            actor = st.selectbox("Examinar un actor del conjunto 80 %", chosen.actor.tolist())
            selected = roles.loc[actor_values.eq(actor)]
            st.write("**Roles asignados a este actor**")
            bar_counts(selected.actor_role, "actor_role", "actor_roles")
            st.caption("Valores originales de actor_role; requieren revisión contextual y no se emparejan listas automáticamente.")
            show_table(selected[["article_id", "source", "title", "title_en", "actor", "actor_role", "url"]], "evidencia_actor")
            st.caption(f"Registros sin actor: {int(roles.actor.eq('').sum())}; no forman parte del denominador del 80 %.")

        st.divider()
        st.subheader("Todos los actores y roles · sin corte del 80 %")
        st.caption("Se incluyen todos los valores dentro de los filtros de medio e idioma. Cada fila aporta una asociación entre el valor de actor y el de actor_role. Las listas o descripciones compuestas se conservan como etiquetas completas: no se inventan emparejamientos.")
        full = roles.copy()
        full["actor_analizado"] = actor_values
        valid_pairs = full.loc[full.actor_analizado.ne("") & full.actor_role.ne("")]
        a, b, c, d = st.columns(4)
        a.metric("Actores únicos", full.loc[full.actor_analizado.ne(""), "actor_analizado"].nunique())
        b.metric("Roles únicos", full.loc[full.actor_role.ne(""), "actor_role"].nunique())
        c.metric("Titulares con actor y rol", len(valid_pairs))
        d.metric("Pares actor–rol distintos", len(valid_pairs[["actor_analizado", "actor_role"]].drop_duplicates()))
        st.caption(f"Sin actor: {int(full.actor_analizado.eq('').sum())} · Sin rol: {int(full.actor_role.eq('').sum())}. Los cruces excluyen pares incompletos; las frecuencias individuales conservan los valores disponibles.")
        for column, label in [("actor_analizado", "Actor"), ("actor_role", "Rol")]:
            counts = full.loc[full[column].ne(""), column].value_counts().rename_axis(label).reset_index(name="articulos")
            if not counts.empty:
                counts["porcentaje"] = counts.articulos / counts.articulos.sum() * 100
                st.write(f"**Frecuencias completas: {label}**")
                fig = px.bar(counts, x=label, y="articulos", hover_data=["porcentaje"])
                st.plotly_chart(fig, width="stretch", key=f"all_{column}")
                show_table(counts, f"frecuencias_completas_{column}")
        if valid_pairs.empty:
            st.info("No hay asociaciones actor–rol completas para construir la matriz.")
        else:
            matrix = pd.crosstab(valid_pairs.actor_analizado, valid_pairs.actor_role)
            matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).index,
                                matrix.sum(axis=0).sort_values(ascending=False).index]
            st.write("**Matriz de coincidencias: actor × rol**")
            st.caption("Es una tabla de contingencia, no una matriz de confusión: no existen etiquetas de referencia para medir aciertos o errores. Un cero significa que la asociación no aparece en la selección.")
            mode = st.radio("Mostrar matriz como", ["Recuentos", "% por actor", "% por rol"], horizontal=True)
            if mode == "% por actor":
                view = matrix.div(matrix.sum(axis=1), axis=0) * 100
            elif mode == "% por rol":
                view = matrix.div(matrix.sum(axis=0), axis=1) * 100
            else:
                view = matrix
            st.caption("% por actor: distribución de roles entre artículos de cada actor con rol disponible. % por rol: distribución de actores entre artículos de cada rol con actor disponible.")
            focus_actors = st.multiselect("Actores de la matriz (vacío = todos)", matrix.index.tolist())
            focus_roles = st.multiselect("Roles de la matriz (vacío = todos)", matrix.columns.tolist())
            visible = view.loc[focus_actors or matrix.index.tolist(), focus_roles or matrix.columns.tolist()]
            fig = px.imshow(visible, aspect="auto", color_continuous_scale="Blues",
                            labels={"x": "Rol", "y": "Actor", "color": mode})
            fig.update_layout(height=max(400, min(1400, 24 * len(visible))))
            st.plotly_chart(fig, width="stretch", key="all_actor_role_heatmap")
            st.caption("Los selectores solo cambian la vista; los porcentajes mantienen el denominador completo de los filtros globales.")
            show_table(view.rename_axis("actor").reset_index(), "matriz_completa_actores_roles")
            combinations = valid_pairs.groupby(["actor_analizado", "actor_role"]).size().rename("articulos").reset_index()
            combinations["porcentaje_del_actor"] = combinations.articulos / combinations.groupby("actor_analizado").articulos.transform("sum") * 100
            combinations["porcentaje_del_rol"] = combinations.articulos / combinations.groupby("actor_role").articulos.transform("sum") * 100
            st.write("**Todas las asociaciones y sus proporciones**")
            show_table(combinations.sort_values("articulos", ascending=False), "asociaciones_actores_roles")
            st.write("**Diversidad de roles por actor**")
            summary = matrix.sum(axis=1).rename("articulos_con_rol").to_frame()
            summary["roles_distintos"] = matrix.gt(0).sum(axis=1)
            summary["rol_mas_frecuente"] = matrix.apply(lambda row: " | ".join(row.index[row.eq(row.max())]), axis=1)
            summary["porcentaje_rol_principal"] = matrix.max(axis=1) / matrix.sum(axis=1) * 100
            st.caption("Se muestran todos los roles empatados como más frecuentes. La diversidad describe variación de etiquetas; por sí sola no demuestra contradicción narrativa.")
            show_table(summary.rename_axis("actor").reset_index(), "diversidad_roles_actores")
            st.write("**Explorar cualquier actor o rol y sus textos**")
            inspect_actors = st.multiselect("Filtrar evidencia por actor", sorted(full.loc[full.actor_analizado.ne(""), "actor_analizado"].unique()))
            inspect_roles = st.multiselect("Filtrar evidencia por rol", sorted(full.loc[full.actor_role.ne(""), "actor_role"].unique()))
            evidence = full
            if inspect_actors:
                evidence = evidence.loc[evidence.actor_analizado.isin(inspect_actors)]
            if inspect_roles:
                evidence = evidence.loc[evidence.actor_role.isin(inspect_roles)]
            show_table(evidence[["article_id", "source", "language", "title", "actor", "actor_analizado", "actor_role", "url"]], "evidencia_todos_actores_roles")

    with tabs[2]:
        st.subheader("Atención temática y diversidad de cobertura")
        thematic = clusters.loc[clusters.cluster.ge(0)]
        pairs = thematic[["article_id", "cluster", "source", "language"]].drop_duplicates()
        if thematic.empty:
            st.info("No hay clusters temáticos disponibles con los filtros actuales.")
        else:
            summary = thematic.groupby("cluster").agg(fragmentos=("fragment_id", "size"), articulos=("article_id", "nunique"))
            summary["medios"] = thematic.loc[thematic.source.ne("")].groupby("cluster").source.nunique()
            summary["idiomas"] = thematic.loc[thematic.language.ne("")].groupby("cluster").language.nunique()
            summary[["medios", "idiomas"]] = summary[["medios", "idiomas"]].fillna(0).astype(int)
            largest_article = thematic.groupby(["cluster", "article_id"]).size().groupby(level=0).max()
            summary["pct_fragmentos_articulo_principal"] = largest_article / summary.fragmentos * 100
            per_source = pairs.loc[pairs.source.ne("")].groupby(["cluster", "source"]).size()
            known_denominator = pairs.loc[pairs.source.ne("")].groupby("cluster").size()
            summary["pct_articulos_medio_principal_entre_fuentes_conocidas"] = per_source.groupby(level=0).max() / known_denominator * 100
            show_table(summary.reset_index().round(2), "diversidad_tematica")
            st.plotly_chart(px.bar(summary.reset_index(), x="cluster", y="articulos"), width="stretch")
            st.write("**Cobertura temática por medio**")
            known_pairs = pairs.loc[pairs.source.ne("")]
            counts = pd.crosstab(known_pairs.source, known_pairs.cluster)
            denominator = clusters.loc[clusters.source.ne("")].groupby("source").article_id.nunique()
            matrix_chart(counts, "topic_coverage", row_label="medio", col_label="cluster", coverage_denominator=denominator)
            breadth = pairs.groupby("article_id").cluster.nunique().rename("numero_de_temas").reset_index()
            show_table(breadth.merge(metadata, on="article_id", how="left"), "amplitud_por_articulo")
            st.write("**Coincidencia de agendas temáticas**")
            binary = counts.gt(0).astype(int)
            if len(binary):
                intersection = binary.to_numpy() @ binary.to_numpy().T
                totals = binary.sum(axis=1).to_numpy()
                union = totals[:, None] + totals[None, :] - intersection
                similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union > 0)
                matrix_chart(pd.DataFrame(similarity * 100, index=binary.index, columns=binary.index), "agenda_similarity", unit="Jaccard (%)", selectable=False)
                st.caption("Jaccard (%) sobre conjuntos de clusters presentes. Mide coincidencia temática, no afinidad narrativa ni ideológica.")
            st.write("**Encuadre del titular × tema del cuerpo**")
            frame_map, _, frame_conflicts = unique_records(articles[["article_id", "frame_analysis"]], "article_id")
            related = pairs[["article_id", "cluster"]].merge(frame_map, on="article_id", validate="many_to_one")
            if frame_conflicts:
                st.caption(f"Solo en este cruce: {frame_conflicts} IDs con encuadres contradictorios excluidos. Las barras conservan todas las filas originales.")
            related = related.loc[related.frame_analysis.ne("")]
            matrix_chart(pd.crosstab(related.frame_analysis, related.cluster), "headline_body", row_label="encuadre", col_label="cluster")
            st.caption("Artículos por combinación; un artículo puede aportar a varios temas. Esta asociación no demuestra contradicción entre titular y cuerpo.")


    with tabs[3]:
        st.subheader("Informes de análisis narrativo")
        st.info("Los informes corresponden a los archivos originales completos: no se filtran automáticamente por medio o idioma. No se cuentan sus menciones como cuotas de narrativas.")
        report_bytes = {}
        if reports_folder.is_dir():
            try:
                for path in reports_folder.iterdir():
                    if path.is_file() and (re.fullmatch(r"cluster_-?\d+_part_\d+\.md", path.name, re.I) or path.name == "globalpulse_report.txt"):
                        try:
                            report_bytes[path.name] = path.read_bytes()
                        except OSError as exc:
                            issues.append(f"No se pudo leer {path.name}: {exc}")
            except OSError as exc:
                issues.append(f"No se pudo abrir la carpeta de informes: {exc}")
        report_bytes.update({k: v for k, v in uploaded.items() if k.lower().endswith((".md", ".txt"))})
        def decode(raw):
            try:
                return raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                return raw.decode("cp1252", errors="replace")
        overview = report_bytes.get("globalpulse_report.txt")
        if overview is not None:
            with st.expander("Informe general · globalpulse_report.txt", expanded=True):
                st.markdown(decode(overview), unsafe_allow_html=False)
                st.download_button("Descargar informe general", overview, "globalpulse_report.txt")
        else:
            st.caption("globalpulse_report.txt no está disponible en la ubicación seleccionada.")
        parts = []
        for name in report_bytes:
            match = re.fullmatch(r"cluster_(-?\d+)_part_(\d+)\.md", name, re.I)
            if match:
                parts.append((int(match[1]), int(match[2]), name))
        if parts:
            parts.sort()
            report_cluster = st.selectbox("Cluster del informe", sorted({p[0] for p in parts}))
            available = [p for p in parts if p[0] == report_cluster]
            selected_name = st.selectbox("Parte del análisis", [p[2] for p in available])
            content = decode(report_bytes[selected_name])
            st.caption("Análisis del lote indicado, normalmente de hasta 12 fragmentos. No equivale automáticamente al análisis completo del cluster.")
            st.markdown(content, unsafe_allow_html=False)
            st.download_button("Descargar esta parte", report_bytes[selected_name], selected_name)
            st.caption("Las referencias numéricas del informe se muestran como texto: no se enlazan a fragment_id sin verificar su correspondencia.")
        else:
            st.info("No hay archivos cluster_N_part_M.md disponibles. Puede subirlos o seleccionar su carpeta.")

    with tabs[4]:
        st.subheader("Qué se calcula y qué queda pendiente")
        st.markdown("""
- Encuadres y tonos: `test_titles.csv`. Actores y roles de titulares: `final_test.csv`. Totales: `gold_titles.csv` y `gold_texts.csv`. Procedencia y temas: `texts.csv` y `clusters_test.csv`.
- Se conservan todas las filas de test_titles y final_test en sus respectivos recuentos. No se cruzan entre sí ni se eliminan titulares por IDs repetidos.
- Las tablas calculadas contienen recuentos, proporciones y cruces derivados; no representan campos nuevos existentes en los archivos originales.
- Los encuadres y tonos se agrupan por su texto exacto. Si contienen descripciones libres, no se consolidan semánticamente.
- No se infieren país, ideología, intención editorial, alcance de audiencia ni opinión pública.
- Sin fechas no hay evolución temporal. Sin audiencia no hay impacto o alcance comercial.
- Los clusters representan agrupaciones temáticas; no se consideran etiquetas de narrativa.
- Los informes se exploran cualitativamente: no permiten calcular cuotas de narrativas en los cuerpos.
- No se comprueba automáticamente la equivalencia entre números de fragmento de los informes y `fragment_id`.
- El corte de actores conserva al menos el 80 % de registros con actor no vacío, incluyendo empates. Solo agrupa las variantes explícitas de United States si se activa esa opción; no valida nombres ni roles.
- Los recuentos describen este corpus, no toda la producción de cada medio. La ausencia de cobertura aquí no demuestra una omisión deliberada.
""")
        st.write("**Disponibilidad y cruces de los archivos**")
        for issue in issues:
            st.write(f"• {issue}")

    with tabs[5]:
        st.subheader("Estadísticas por idioma original")
        st.caption("Distribución del corpus de artículos disponibles en gold_texts.csv según language, sin inferir idiomas. Los vacíos se muestran como Sin dato.")
        data = tables["gold_texts.csv"].copy()
        if selected_languages:
            data = data.loc[data.language.isin(selected_languages)]
        if selected_sources:
            source_ids = texts.loc[texts.source.isin(selected_sources), "article_id"]
            data = data.loc[data.article_id.ne("") & data.article_id.isin(source_ids)]
        counts = data.language.replace("", "Sin dato").value_counts().rename_axis("idioma").reset_index(name="articulos")
        st.write("**Artículos disponibles por idioma**")
        if counts.empty:
            st.info("No hay artículos disponibles en esta selección de gold_texts.csv.")
        else:
            counts["porcentaje"] = counts.articulos / counts.articulos.sum() * 100
            st.plotly_chart(px.bar(counts, x="idioma", y="articulos", hover_data=["porcentaje"]), width="stretch", key="language_corpus")
            show_table(counts, "idiomas_corpus")
        st.caption("Una fila de gold_texts.csv cuenta como un artículo. Se aplican los filtros globales; los indicadores superiores conservan los totales completos.")
        for col in ["frame_analysis", "tone"]:
            st.write(f"**Idioma × {col} · test_titles.csv**")
            data = articles.loc[articles[col].ne("")].copy()
            data["language"] = data.language.replace("", "Sin dato")
            matrix_chart(pd.crosstab(data.language, data[col]), f"language_{col}", percent=True, unit="Titulares", row_label="idioma", col_label="encuadre" if col == "frame_analysis" else "tono")
        st.write("**Idioma × rol · final_test.csv**")
        data = roles.loc[roles.actor_role.ne("")].copy()
        data["language"] = data.language.replace("", "Sin dato")
        matrix_chart(pd.crosstab(data.language, data.actor_role), "language_roles", percent=True, unit="Titulares", row_label="idioma", col_label="rol")

    with tabs[6]:
        st.subheader("Red de similitud temática entre clusters")
        st.caption("Tamaño del nodo: artículos únicos. Grosor del enlace: similitud coseno entre centroides de embeddings normalizados. La disposición circular no representa distancias semánticas. Se excluye el ruido (-1).")
        thematic = clusters.loc[clusters.cluster.ge(0)].copy()
        options = thematic.groupby("cluster").article_id.nunique().sort_values(ascending=False).index.tolist()
        chosen = st.multiselect("Clusters de la red", options, default=options[:30], key="network_clusters")
        threshold = st.slider("Similitud mínima entre clusters", 0.0, 1.0, 0.65, 0.01, key="network_similarity")
        neighbours = st.slider("Máximo de vecinos elegidos por cluster", 1, 15, 3, key="network_neighbours")
        st.caption("Se inicia con hasta 30 clusters de mayor cobertura. Un enlace aparece si cualquiera de sus extremos lo selecciona entre sus vecinos; el grado final puede superar ese máximo.")
        selected = thematic.loc[thematic.cluster.isin(chosen)]
        embedding_source = st.selectbox("Archivo de embeddings de la red", ["clusters_test.csv", "embbedings_test.csv"], key="network_embedding_source")
        st.caption("Se usa un único archivo de vectores, enlazado por article_id + fragment_id. Selecciona embbedings_test.csv si conserva los vectores completos. Ambos archivos deben corresponder a la misma ejecución y modelo; no se mezclan automáticamente.")
        vectors = tables[embedding_source][["article_id", "fragment_id", "embedding"]].drop_duplicates()
        conflicting = vectors.duplicated(["article_id", "fragment_id"], keep=False)
        conflict_keys = set(map(tuple, vectors.loc[conflicting, ["article_id", "fragment_id"]].values))
        vectors = vectors.loc[~conflicting]
        selected = selected.merge(vectors, on=["article_id", "fragment_id"], how="left", validate="one_to_one", indicator=True)
        parsed = selected.embedding.map(parse_embedding)
        dimensions = parsed.map(lambda v: len(v) if v is not None else 0)
        invalid = int(dimensions.eq(0).sum())
        valid_dims = sorted(dimensions.loc[dimensions.gt(0)].unique().tolist())
        def reason(row):
            if (row.article_id, row.fragment_id) in conflict_keys:
                return "Clave con embeddings contradictorios"
            if row._merge == "left_only":
                return "Sin correspondencia por identificadores"
            raw = row.embedding
            if not isinstance(raw, str) or not raw.strip():
                return "Embedding vacío"
            if "..." in raw or "…" in raw:
                return "Vector truncado (contiene puntos suspensivos)"
            if parse_embedding(raw) is None:
                return "Formato inválido, vector nulo o valores no finitos"
            return "Válido"
        diagnostic = selected.apply(reason, axis=1) if len(selected) else pd.Series(dtype=str)
        st.caption(f"Fragmentos sin embedding válido: {invalid} de {len(selected)}.")
        if len(selected):
            st.dataframe(diagnostic.value_counts().rename_axis("diagnostico").reset_index(name="fragmentos"), hide_index=True)
        if invalid:
            st.warning("Los vectores escritos como [0.1 0.2 -0.3] ahora se admiten. Si contienen '...', faltan componentes y deben exportarse completos desde el pipeline; el dashboard no puede reconstruirlos.")
            sample = selected.loc[dimensions.eq(0), ["article_id", "fragment_id", "embedding"]].head(5).copy()
            sample["diagnostico"] = diagnostic.loc[sample.index]
            sample["embedding"] = sample.embedding.fillna("").str.slice(0, 300)
            st.write("Muestra del texto recibido (vista limitada a 300 caracteres por celda)")
            st.dataframe(sample, hide_index=True)
        centroids = {}
        if valid_dims:
            dimension = st.selectbox("Dimensión del embedding", valid_dims, key="network_dimension")
            eligible = selected.loc[dimensions.eq(dimension)].copy()
            eligible["vector"] = parsed.loc[dimensions.eq(dimension)]
            for cluster_id, group in eligible.groupby("cluster"):
                center = np.stack(group.vector.tolist()).mean(axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    centroids[cluster_id] = center / norm
            st.caption(f"Embeddings válidos de otras dimensiones excluidos: {int((dimensions.gt(0) & dimensions.ne(dimension)).sum())}.")
        keys = sorted(centroids)
        edge_pairs = set()
        if keys:
            centers = np.stack([centroids[k] for k in keys])
            similarities = np.clip(centers @ centers.T, -1, 1)
            for i in range(len(keys)):
                candidates = [j for j in range(len(keys)) if j != i and similarities[i,j] >= threshold]
                for j in sorted(candidates, key=lambda j: (-similarities[i,j], keys[j]))[:neighbours]:
                    edge_pairs.add(tuple(sorted((i,j))))
        sizes = selected.groupby("cluster").article_id.nunique()
        nodes = [dict(id=k,label=f"Cluster {k}",group="Cluster",size=int(sizes[k]),
                      hover=f"Cluster {k} · {sizes[k]} artículos únicos · centro calculado con embeddings válidos") for k in keys]
        edges = [dict(left=keys[i],right=keys[j],weight=float(similarities[i,j]),
                      hover=f"Cluster {keys[i]} ↔ {keys[j]} · coseno {similarities[i,j]:.3f}") for i,j in sorted(edge_pairs)]
        network_plot(nodes,edges,"cluster_network")
        st.caption(f"Clusters seleccionados sin centro válido: {len(set(chosen)-set(keys))}. La similitud temática no demuestra una narrativa compartida.")
        if edges:
            show_table(pd.DataFrame([dict(cluster_a=e["left"],cluster_b=e["right"],similitud_coseno=e["weight"]) for e in edges]),"enlaces_clusters")

    with tabs[7]:
        st.subheader("Red de actores y roles en los titulares")
        st.caption("Dos grupos de nodos: actores y roles. Un enlace representa una asociación registrada en final_test.csv. Su grosor y los tamaños de nodos reflejan filas, sin interpretar sentimiento ni corregir identificaciones ambiguas.")
        network_data = roles.copy()
        network_data["actor_red"] = actor_values
        use_pareto = st.checkbox("Mostrar actores del conjunto acumulado del 80 %", value=True, key="network_actor_pareto")
        if use_pareto:
            allowed = ranking.loc[ranking.incluido, "actor"]
            network_data = network_data.loc[network_data.actor_red.isin(allowed)]
        network_data = network_data.loc[network_data.actor_red.ne("") & network_data.actor_role.ne("")]
        chosen_actors = st.multiselect("Actores de la red (vacío = todos)", sorted(network_data.actor_red.unique()), key="network_actors")
        chosen_roles = st.multiselect("Roles de la red (vacío = todos)", sorted(network_data.actor_role.unique()), key="network_roles")
        if chosen_actors:
            network_data = network_data.loc[network_data.actor_red.isin(chosen_actors)]
        if chosen_roles:
            network_data = network_data.loc[network_data.actor_role.isin(chosen_roles)]
        links = network_data.groupby(["actor_red", "actor_role"]).size().reset_index(name="registros")
        minimum = st.number_input("Mínimo de registros por enlace", min_value=1, value=1, step=1, key="network_min_links")
        links = links.loc[links.registros.ge(minimum)]
        nodes = []
        for field,group,prefix in [("actor_red","Actor","actor:"),("actor_role","Rol","role:")]:
            for label,total in links.groupby(field).registros.sum().items():
                nodes.append(dict(id=prefix+label,label=label,group=group,size=int(total),hover=f"{group}: {label} · {total} registros en enlaces visibles"))
        edges = [dict(left="actor:"+r.actor_red,right="role:"+r.actor_role,weight=int(r.registros),
                      hover=f"{r.actor_red} → {r.actor_role} · {r.registros} registros") for r in links.itertuples()]
        network_plot(nodes,edges,"actor_role_network",bipartite=True)
        st.caption("Se aplica la misma normalización opcional de Estados Unidos y el mismo conjunto del 80 % de la pestaña Actores y roles. Las etiquetas compuestas se conservan enteras. El filtro de frecuencia no valida actores.")
        if not links.empty:
            show_table(links.rename(columns={"actor_red":"actor"}),"enlaces_actores_roles")

    with tabs[8]:
        st.subheader("Inteligencia narrativa · análisis de los cuerpos")
        st.info("Contenido cualitativo de los informes existentes, organizado sin nuevas llamadas al LLM. Cada conclusión pertenece al lote y al archivo indicado; no representa una cuota narrativa del corpus ni una validación de sus afirmaciones.")
        st.caption("Los filtros globales de medio e idioma no se aplican a los informes. Las citas y atribuciones se muestran tal como fueron generadas; las referencias numéricas no se enlazan automáticamente con fragment_id porque su equivalencia no está verificada.")
        scope = st.radio("Explorar", ["Análisis por cluster y parte", "Informe general"], horizontal=True, key="narrative_scope")
        selected_reports = []
        if scope == "Informe general":
            if overview is None:
                st.info("No está disponible globalpulse_report.txt. Selecciona la carpeta o súbelo desde la barra lateral.")
            else:
                selected_reports = [(None, None, "globalpulse_report.txt")]
        elif not parts:
            st.info("No hay archivos cluster_N_part_M.md. Selecciona la carpeta cluster_analysis o sube sus informes.")
        else:
            cluster_id = st.selectbox("Cluster para explorar narrativas", sorted({p[0] for p in parts}), key="narrative_cluster")
            available_parts = [p for p in parts if p[0] == cluster_id]
            names = [p[2] for p in available_parts]
            selected_names = st.multiselect("Partes del cluster", names, default=names[:1], key="narrative_parts")
            selected_reports = [p for p in available_parts if p[2] in selected_names]
            st.caption("Selecciona varias partes para contrastarlas. Los argumentos contrapuestos se conservan dentro de los hallazgos originales; no se generan nuevas interpretaciones.")
        categories = ["Tema principal", "Subtemas", "Narrativas", "Hallazgos y argumentos", "Citas y atribuciones", "Otras secciones"]
        selected_categories = st.multiselect("Secciones a mostrar", categories, default=categories, key="narrative_categories")
        query = st.text_input("Buscar texto literal en los análisis seleccionados", key="narrative_search").strip()
        catalog = []
        for cluster_id, part_id, filename in selected_reports:
            original = decode(report_bytes[filename])
            sections = split_report_sections(original)
            st.write(f"**{filename}**")
            if cluster_id is not None:
                st.caption(f"Cluster {cluster_id} · parte {part_id} · análisis de un lote, normalmente de hasta 12 fragmentos.")
            else:
                st.caption("Informe general: se conserva su estructura original; no se deducen correspondencias con clusters a partir del texto.")
            visible = [section for section in sections if section["categoria"] in selected_categories
                       and (not query or query.casefold() in (section["titulo"] + "\n" + section["texto"]).casefold())]
            if not visible:
                st.info("No hay secciones coincidentes con esta selección.")
            for index, section in enumerate(visible):
                with st.expander(f'{section["categoria"]} · {section["titulo"]}', expanded=True):
                    st.caption(f"Procedencia: {filename}")
                    st.markdown(section["texto"], unsafe_allow_html=False)
                catalog.append(dict(archivo=filename, cluster=cluster_id, parte=part_id, **section))
            with st.expander(f"Consultar original completo · {filename}"):
                st.markdown(original, unsafe_allow_html=False)
                st.download_button("Descargar informe original", report_bytes[filename], filename, key=f"narrative_original_{filename}")
            st.caption("Las secciones no reconocidas se conservan en Otras secciones. Esta vista no recupera contenido ausente o truncado en el archivo original.")
        if catalog:
            export_data = pd.DataFrame(catalog)
            st.download_button("Descargar secciones visibles", export_data.to_csv(index=False).encode("utf-8-sig"),
                               "inteligencia_narrativa.csv", mime="text/csv", key="narrative_export")


if __name__ == "__main__":
    main()
