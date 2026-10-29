"""GlobalPulse — ejecutar: python -m streamlit run dashboard_pipeline.py"""
from pathlib import Path
from io import BytesIO
import ast
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

SCHEMAS = {
 'urls.csv': 'title url source language',
 'titles_to_translate.csv': 'language title article_id',
 'texts_to_translate.csv': 'title language body article_id',
 'texts.csv': 'title url source language body extraction_status extraction_method article_id',
 'test_titles.csv': 'title article_id language title_en frame_analysis tone',
 'gold_titles.csv': 'title article_id language title_en',
 'gold_texts.csv': 'body article_id title language body_en title_en',
 'fragments_test.csv': 'article_id fragment_id fragment',
 'final_test.csv': 'title article_id language title_en actor actor_role',
 'embbedings_test.csv': 'article_id fragment_id fragment embedding',
 'clusters_test.csv': 'article_id fragment_id fragment embedding cluster cluster_probability',
}

def present(s):
    return s.fillna('').astype(str).str.strip().ne('')

def ids(df, field='article_id'):
    return set(df.loc[present(df[field]), field]) if field in df else set()

def ratio(a, b):
    return f'{100*a/b:.1f}%' if b else 'N/D'

def field_summary(df, fields):
    """Cuenta todas las filas; no deduplica ni exige ID para evaluar campos."""
    return pd.DataFrame([
        dict(campo=col, filas=len(df), no_vacios=int(present(df[col]).sum()),
             vacios=int((~present(df[col])).sum()),
             cobertura=ratio(int(present(df[col]).sum()),len(df)))
        for col in fields if col in df
    ])

@st.cache_data(show_spinner=False)
def read_csv(raw, separator, encoding):
    df = pd.read_csv(BytesIO(raw), sep=separator, encoding=encoding, dtype=str,
                     keep_default_na=False)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.str.match(r'^Unnamed:')]
    for col in ['article_id', 'fragment_id']:
        if col in df:
            df[col] = df[col].str.strip()
    return df

def chart(fig):
    fig.update_layout(template='plotly_white', margin=dict(l=10,r=10,t=45,b=10))
    st.plotly_chart(fig, width='stretch')

def counts(df, col, title):
    if col in df and len(df):
        d = df[col].replace('', '(vacío)').value_counts().rename_axis(col).reset_index(name='filas')
        chart(px.bar(d.head(20), x=col, y='filas', title=title))

def export(df, name):
    st.download_button('Descargar ' + name, df.to_csv(index=False).encode('utf-8-sig'), name, 'text/csv')

def vector(raw):
    try:
        try:
            obj = json.loads(raw)
        except (ValueError, TypeError):
            obj = ast.literal_eval(raw)
        v = np.asarray(obj, dtype=float)
        return v if v.ndim == 1 and v.size and np.isfinite(v).all() and np.linalg.norm(v)>0 else None
    except (ValueError, TypeError, SyntaxError, MemoryError):
        return None

def main():
    st.set_page_config(page_title='GlobalPulse · Pipeline', page_icon='📊', layout='wide')
    st.title('GlobalPulse | Calidad del pipeline')
    st.caption('Auditoría del corpus cargado · artículos, traducciones, fragmentos y clusters')
    st.sidebar.header('Datos')
    mode = st.sidebar.radio('Origen', ['Carpeta local', 'Subir CSV'])
    separator = st.sidebar.selectbox('Separador CSV', [',', ';', '\t'])
    encoding = st.sidebar.selectbox('Codificación', ['utf-8-sig', 'utf-8', 'cp1252'])
    raw_files = {}
    if mode == 'Carpeta local':
        folder = Path(st.sidebar.text_input('Carpeta de los CSV', str(Path(__file__).resolve().parent))).expanduser()
        for name in SCHEMAS:
            p = folder / name
            if p.is_file():
                try:
                    raw_files[name] = p.read_bytes()
                except OSError as e:
                    st.warning(f'{name}: {e}')
    else:
        for f in st.sidebar.file_uploader('Selecciona tus CSV', type='csv', accept_multiple_files=True) or []:
            if f.name in SCHEMAS:
                raw_files[f.name] = f.getvalue()
            else:
                st.sidebar.warning(f'Nombre no reconocido: {f.name}')
    if st.sidebar.button('Actualizar datos'):
        st.cache_data.clear()
        st.rerun()
    data, inventory, issues = {}, [], []
    for name, schema in SCHEMAS.items():
        if name not in raw_files:
            inventory.append(dict(archivo=name, estado='Ausente', filas=None, columnas_faltantes=''))
            continue
        try:
            d = read_csv(raw_files[name], separator, encoding)
        except Exception as e:
            inventory.append(dict(archivo=name, estado='Error de lectura', filas=None, columnas_faltantes=str(e)))
            continue
        missing = sorted(set(schema.split()) - set(d.columns))
        inventory.append(dict(archivo=name, estado='Incompleto' if missing else 'OK', filas=len(d), columnas_faltantes=', '.join(missing)))
        # Se retiene el esquema real para no confundir una columna ausente con valores vacíos.
        d.attrs['original_columns'] = set(d.columns)
        for col in missing:
            d[col] = ''
        data[name] = d
        for field in ['article_id', 'fragment_id']:
            if field in schema.split():
                n = int((~present(d[field])).sum())
                if n: issues.append(dict(archivo=name, control=f'{field} vacío', filas=n))
        key = ['article_id','fragment_id'] if 'fragment_id' in schema.split() else (['article_id'] if 'article_id' in schema.split() else ['url'])
        valid = d[key].apply(present).all(axis=1)
        dup = int(d.loc[valid].duplicated(key).sum())
        if dup: issues.append(dict(archivo=name, control='Claves duplicadas (filas extra)', filas=dup))
    if not data:
        st.info('Coloca los CSV en la carpeta indicada o súbelos. Se admiten archivos parciales; no se generan resultados ficticios.')
        st.dataframe(pd.DataFrame(inventory), width='stretch')
        return
    def get(name):
        return data.get(name, pd.DataFrame(columns=SCHEMAS[name].split()))
    def available(name, *cols):
        return name in data and set(cols).issubset(data[name].attrs['original_columns'])
    t, u, g, f, c = [get(n) for n in ['texts.csv','urls.csv','gold_texts.csv','fragments_test.csv','clusters_test.csv']]
    st.sidebar.header('Criterios de revisión')
    min_chars = st.sidebar.number_input('Artículo corto: caracteres mínimos', 0, 100000, 300)
    weak = st.sidebar.slider('Pertenencia débil: umbral', 0.0, 1.0, 0.5)
    status_values = sorted(t.extraction_status.unique()) if len(t) else []
    defaults = [x for x in status_values if x.strip().lower() in ['success','ok','successful','extracted','true','1']]
    success = st.sidebar.multiselect('Estados de extracción exitosa', status_values, default=defaults)
    st.sidebar.caption('Selecciona los estados que realmente indican éxito en tu extractor. Los umbrales son reglas de revisión.')
    tabs = st.tabs(['Resumen y etapas', 'Extracción', 'Traducción', 'Fragmentos y clusters', 'Integridad y datos', 'Análisis de titulares'])
    with tabs[0]:
        left,right = st.columns(2)
        left.metric('Titulares disponibles · filas de gold_titles',len(get('gold_titles.csv')) if 'gold_titles.csv' in data else 'N/D')
        right.metric('Artículos disponibles · filas de gold_texts',len(g) if 'gold_texts.csv' in data else 'N/D')
        st.caption('Totales por filas, sin deduplicar. Las coberturas por article_id se muestran por separado y no eliminan registros de los archivos.')
        a,b,d,e = st.columns(4)
        a.metric('URLs únicas', u.loc[present(u.url),'url'].nunique() if 'urls.csv' in data else 'N/D')
        b.metric('Fuentes', u.loc[present(u.source),'source'].nunique() if 'urls.csv' in data else 'N/D')
        d.metric('Artículos con cuerpo', len(ids(t.loc[present(t.body)])) if 'texts.csv' in data else 'N/D')
        e.metric('Fragmentos (filas)', len(f) if 'fragments_test.csv' in data else 'N/D')
        stages = [('texts.csv','Extraídos con cuerpo','body'), ('gold_texts.csv','Con body_en','body_en'),
                  ('fragments_test.csv','Fragmentados','fragment'),('embbedings_test.csv','Con embedding','embedding'),
                  ('clusters_test.csv','Con etiqueta de cluster','cluster')]
        base = ids(t)
        rows=[]
        for name,label,col in stages:
            if available(name, col, 'article_id'):
                s = ids(get(name).loc[present(get(name)[col])])
                rows.append(dict(etapa=label, articulos=len(s), en_texts=len(s & base) if available('texts.csv','article_id') else None,
                                 fuera_de_texts=len(s-base) if available('texts.csv','article_id') else None))
        if rows:
            stage_df=pd.DataFrame(rows)
            chart(px.bar(stage_df, x='etapa', y='articulos', title='Artículos únicos por etapa · sin restringir a texts.csv'))
            st.dataframe(stage_df, width='stretch', hide_index=True)
        st.caption('Etapas independientes del cuerpo del artículo: una subida o bajada no implica pérdida causal. Una etiqueta de cluster presente no garantiza validez; se revisa en su pestaña. Los titulares se auditan aparte. No se incluyen las colas de traducción como pasos obligatorios. Usa archivos de una misma ejecución.')
        # urls no tiene article_id: solo se enlaza por URL exacta.
        if 'urls.csv' in data and 'texts.csv' in data:
            urls = set(u.loc[present(u.url),'url'])
            seen = set(t.loc[present(t.url),'url'])
            st.metric('URLs de búsqueda presentes en texts', ratio(len(urls & seen), len(urls)))
            st.caption('Cruce por URL exacta; no se usa el titular como identificador.')
        counts(u,'source','Medios encontrados · top 20')
        counts(u,'language','Idiomas encontrados')
        st.info('Duración, throughput, coste/API e histórico: N/D. Requieren tiempos, tokens, costes y run_id registrados por el pipeline.')
    with tabs[1]:
        if 'texts.csv' not in data: st.info('Falta texts.csv.')
        else:
            valid_body=present(t.body)
            ok=t.extraction_status.isin(success)
            a,b,d=st.columns(3)
            a.metric('Éxito por estado (filas)', ratio(int(ok.sum()),len(t)) if success else 'Configura estados')
            b.metric('Cuerpos no vacíos',ratio(int(valid_body.sum()),len(t)))
            d.metric('Cuerpos cortos no vacíos',int((valid_body & t.body.str.len().lt(min_chars)).sum()))
            counts(t,'extraction_status','Estados de extracción')
            counts(t,'extraction_method','Método final registrado')
            q=t.assign(caracteres=t.body.str.len(), cuerpo_presente=valid_body, estado_exitoso=ok)
            chart(px.histogram(q,x='caracteres',nbins=40,title='Longitud de los cuerpos · caracteres'))
            if success:
                stats=q.groupby('extraction_method',dropna=False).agg(filas=('body','size'), exitos=('estado_exitoso','sum'),con_cuerpo=('cuerpo_presente','sum')).reset_index()
                stats['porcentaje_estado_exitoso']=100*stats.exitos/stats.filas
                chart(px.bar(stats,x='extraction_method',y='porcentaje_estado_exitoso',range_y=[0,100],hover_data=['filas'],title='Éxito observado según método final'))
                st.caption('El método final no registra los intentos previos: esta comparación no mide la eficacia de cada intento ni controla la dificultad de los sitios.')
            bad=q.loc[~valid_body | q.caracteres.lt(min_chars) | (ok & ~valid_body)]
            st.write('Artículos para revisar')
            st.dataframe(bad.drop(columns=['embedding'],errors='ignore'),width='stretch')
            export(bad,'extracciones_a_revisar.csv')
    with tabs[2]:
        rows=[]
        for name,col in [('gold_titles.csv','title_en'),('gold_texts.csv','body_en'),('gold_texts.csv','title_en')]:
            if available(name,col):
                d=get(name)
                rows.append(dict(archivo=name,campo=col,filas=len(d),no_vacias=int(present(d[col]).sum()),cobertura=ratio(int(present(d[col]).sum()),len(d))))
        st.dataframe(pd.DataFrame(rows),width='stretch')
        st.caption('Cobertura significa texto presente; no evalúa fidelidad de la traducción. Un original inglés puede conservarse idéntico.')
        for pending,done,col in [('titles_to_translate.csv','gold_titles.csv','title_en'),('texts_to_translate.csv','gold_texts.csv','body_en')]:
            if available(pending,'article_id') and available(done,'article_id',col):
                queued=ids(get(pending)); completed=ids(get(done).loc[present(get(done)[col])])
                st.metric(f'Cobertura de {pending}',ratio(len(queued & completed),len(queued)))
                st.caption(f'{len(queued-completed)} article_id de la cola sin salida no vacía en {done}.')
        if len(g):
            lengths=g.assign(original=g.body.str.len(),ingles=g.body_en.str.len())
            chart(px.scatter(lengths,x='original',y='ingles',hover_data=['article_id','language'],title='Longitud original / inglés · caracteres'))
    with tabs[3]:
        if len(f):
            counts(f.assign(longitud=f.fragment.str.len()),'article_id','Fragmentos por artículo · top 20')
            chart(px.histogram(f.assign(caracteres=f.fragment.str.len()),x='caracteres',nbins=40,title='Longitud de los fragmentos'))
            st.metric('Fragmentos repetidos (filas extra)',int(f.loc[present(f.fragment)].duplicated('fragment').sum()))
        if len(c):
            c=c.copy(); labels=pd.to_numeric(c.cluster,errors='coerce'); prob=pd.to_numeric(c.cluster_probability,errors='coerce')
            valid=labels.notna() & np.isfinite(labels) & labels.ge(-1) & labels.eq(labels.round())
            assigned=valid & labels.ge(0); noise=valid & labels.eq(-1)
            a,b,d=st.columns(3)
            a.metric('Clusters',labels[assigned].nunique()); b.metric('Ruido / etiquetas válidas',ratio(int(noise.sum()),int(valid.sum())))
            d.metric('Etiquetas inválidas/especiales',int((~valid).sum()))
            st.caption('Se considera ruido -1. Otros códigos negativos o valores inválidos se muestran aparte. La pertenencia no es una probabilidad calibrada de calidad semántica.')
            c['probabilidad']=prob; c['cluster_num']=labels
            st.metric('Probabilidades inválidas',int((prob.isna() | ~np.isfinite(prob) | ~prob.between(0,1)).sum()))
            st.metric('Asignados con pertenencia débil',int((assigned & prob.between(0,weak,inclusive='left')).sum()))
            if assigned.any():
                sizes=c.loc[assigned].groupby('cluster_num').size().reset_index(name='fragmentos')
                sizes['cluster_num']=sizes.cluster_num.astype(int).astype(str)
                chart(px.bar(sizes.sort_values('fragmentos',ascending=False),x='cluster_num',y='fragmentos',title='Tamaño de clusters · filas'))
                box=c.loc[assigned & prob.between(0,1)].copy(); box['cluster_num']=box.cluster_num.astype(int).astype(str)
                chart(px.box(box,x='cluster_num',y='probabilidad',title='Distribución de pertenencia por cluster'))
                diversity=c.loc[assigned].groupby('cluster_num').agg(fragmentos=('fragment_id','size'),articulos=('article_id',lambda s:s[s.ne('')].nunique())).reset_index()
                st.write('Artículos únicos por cluster'); st.dataframe(diversity,hide_index=True)
            st.write('Fragmentos para revisar')
            review=c.loc[noise | ~valid | ~prob.between(0,1) | (assigned & prob.lt(weak))].drop(columns='embedding',errors='ignore')
            st.dataframe(review,width='stretch'); export(review,'fragmentos_a_revisar.csv')
        if st.checkbox('Validar vectores de embeddings (puede tardar)'):
            for name in ['embbedings_test.csv','clusters_test.csv']:
                if name in data:
                    vectors=get(name).embedding.map(vector)
                    dims=vectors.map(lambda v:len(v) if v is not None else 0)
                    st.write(name); st.metric('Vectores inválidos o nulos',int(dims.eq(0).sum()))
                    st.dataframe(dims.value_counts().rename_axis('dimension').reset_index(name='filas'),hide_index=True)
    with tabs[4]:
        st.write('Inventario y esquema'); st.dataframe(pd.DataFrame(inventory),hide_index=True,width='stretch')
        for name,d in data.items():
            if name!='texts.csv' and 'article_id' in d and 'texts.csv' in data:
                n=int((present(d.article_id) & ~d.article_id.isin(ids(t))).sum())
                if n: issues.append(dict(archivo=name,control='article_id ausente en texts.csv',filas=n))
        # Clave compuesta: fragment_id puede reiniciarse en cada artículo.
        for left,right in [('fragments_test.csv','embbedings_test.csv'),('embbedings_test.csv','clusters_test.csv')]:
            if left in data and right in data:
                key=['article_id','fragment_id']; l=get(left); r=get(right)
                ls=set(map(tuple,l.loc[l[key].apply(present).all(axis=1),key].values))
                rs=set(map(tuple,r.loc[r[key].apply(present).all(axis=1),key].values))
                st.write(f'{left} → {right}: {len(ls-rs)} claves sin salida; {len(rs-ls)} claves sin origen.')
        audit=pd.DataFrame(issues,columns=['archivo','control','filas'])
        st.write('Incidencias de integridad'); st.dataframe(audit,hide_index=True); export(audit,'auditoria_pipeline.csv')
        st.caption('Los duplicados se cuentan como filas adicionales. Filas sin identificador se excluyen de los cruces. Diferencias entre archivos pueden indicar ejecuciones mezcladas: falta run_id para comprobarlo.')
        selected=st.selectbox('Explorar archivo',list(data))
        view=data[selected].drop(columns='embedding',errors='ignore')
        query=st.text_input('Buscar texto literal')
        if query:
            view=view.loc[view.astype(str).apply(lambda s:s.str.contains(query,case=False,regex=False)).any(axis=1)]
        st.dataframe(view.head(1000),width='stretch')
        st.caption(f'{len(view):,} filas coincidentes. Vista limitada a 1.000 filas.')

    with tabs[5]:
        st.subheader('Calidad del análisis de titulares')
        st.caption('test_titles.csv aporta encuadres y tonos. final_test.csv aporta actores y roles de los titulares, no del cuerpo. Sus campos repetidos de encuadre/tono no se utilizan.')
        base_name='gold_titles.csv'
        base_titles=get(base_name)
        for name,fields in [('test_titles.csv',['frame_analysis','tone']),
                            ('final_test.csv',['actor','actor_role'])]:
            st.write(f'**{name}**')
            if name not in data:
                st.info(f'Falta {name}.')
                continue
            d=get(name)
            usable=[col for col in fields if available(name,col)]
            missing=[col for col in fields if col not in usable]
            if missing:
                st.warning('Campos ausentes, sin evaluar: '+', '.join(missing))
            st.metric(f'Filas cargadas · {name}',len(d))
            if usable:
                summary=field_summary(d,usable)
                st.dataframe(summary,hide_index=True,width='stretch')
                export(summary,'calidad_'+name)
                for col in usable:
                    # Incluye vacíos y todas las categorías: la suma equivale a las filas.
                    labels=d[col].where(present(d[col]),'(vacío)')
                    distribution=labels.value_counts().rename_axis(col).reset_index(name='filas')
                    chart(px.bar(distribution,x=col,y='filas',title=f'{col} · todas las filas de {name}'))
            if available(base_name,'article_id') and available(name,'article_id'):
                expected=ids(base_titles); observed=ids(d)
                complete=d.loc[d[usable].apply(present).all(axis=1)] if len(usable)==len(fields) else d.iloc[:0]
                complete_ids=ids(complete)
                n_missing=int((~present(d.article_id)).sum())
                st.write(f'IDs únicos en gold_titles: {len(expected)} · En este archivo: {len(observed)} · Coincidentes: {len(expected & observed)} · Fuera de gold_titles: {len(observed-expected)}.')
                st.metric(f'Cobertura de IDs de gold_titles · {name}',ratio(len(expected & observed),len(expected)))
                if len(usable)==len(fields):
                    st.metric(f'IDs de gold_titles con ambos campos presentes · {name}',ratio(len(expected & complete_ids),len(expected)))
                pending=base_titles.loc[present(base_titles.article_id) & ~base_titles.article_id.isin(observed)]
                st.write(f'Filas de gold_titles sin correspondencia en {name}: {len(pending)}. Filas del análisis sin ID: {n_missing}.')
                if len(pending):
                    st.dataframe(pending,width='stretch')
                    export(pending,'sin_correspondencia_'+name)
            st.caption('Las frecuencias conservan todas las filas, incluidas las de ID repetido o vacío. Los cruces usan conjuntos de IDs no vacíos; no multiplican filas. Campos presentes no implica análisis correcto. El corte del 80 % de actores pertenece al dashboard de inteligencia; aquí se auditan todos.')

if __name__=='__main__':
    main()
