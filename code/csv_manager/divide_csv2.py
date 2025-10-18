import pandas as pd
import ast
import re

# Leer el archivo CSV
df = pd.read_csv(r"c:\Users\Usuario\Documents\Workspace\Estudio-Psiquiatricos\datasets\neo4j\diagnosticos_psiquiatricos_por_paciente.csv")

def extraer_diagnosticos_avanzado(valor):
    """Función avanzada para extraer diagnósticos con múltiples formatos"""
    if pd.isna(valor) or str(valor).strip() in ['', '[]']:
        return []
    
    valor_str = str(valor).strip()
    diagnosticos = set()  # Usar set para evitar duplicados
    
    # Método 1: Evaluar como lista Python
    try:
        lista_eval = ast.literal_eval(valor_str)
        if isinstance(lista_eval, list):
            for item in lista_eval:
                if item:
                    item_str = str(item).strip()
                    # Buscar todos los códigos F en el item
                    codigos = re.findall(r'F\d{1,2}(?:\.\d{1,3})?', item_str)
                    diagnosticos.update(codigos)
    except:
        pass
    
    # Método 2: Búsqueda directa con regex
    codigos_regex = re.findall(r'F\d{1,2}(?:\.\d{1,3})?', valor_str)
    diagnosticos.update(codigos_regex)
    
    # Método 3: Manejo de casos especiales como "F20, F21"
    # Primero remover corchetes y comillas
    valor_limpio = re.sub(r'[\[\]\'"]+', '', valor_str)
    # Buscar patrones separados por comas
    if ', ' in valor_limpio:
        partes = valor_limpio.split(', ')
        for parte in partes:
            codigos_parte = re.findall(r'F\d{1,2}(?:\.\d{1,3})?', parte)
            diagnosticos.update(codigos_parte)
    
    return sorted(list(diagnosticos))

# Procesar datos
resultados = []
errores = []

print("🔄 Procesando datos con método avanzado...")

for index, row in df.iterrows():
    numero_historia = row['Nº Historia']
    diag_psq = row['DIAG PSQ']
    
    try:
        diagnosticos = extraer_diagnosticos_avanzado(diag_psq)
        
        if not diagnosticos:
            # Registrar filas sin diagnósticos válidos
            errores.append({
                'Numero_Historia': numero_historia,
                'DIAG_PSQ_Original': diag_psq,
                'Error': 'No se encontraron diagnósticos válidos'
            })
        
        for diagnostico in diagnosticos:
            resultados.append({
                'Numero_Historia': numero_historia,
                'Diagnostico': diagnostico
            })
            
    except Exception as e:
        errores.append({
            'Numero_Historia': numero_historia,
            'DIAG_PSQ_Original': diag_psq,
            'Error': str(e)
        })

# Crear DataFrames
df_resultados = pd.DataFrame(resultados)
df_errores = pd.DataFrame(errores)

# Eliminar duplicados
df_resultados = df_resultados.drop_duplicates().sort_values(['Numero_Historia', 'Diagnostico']).reset_index(drop=True)

# Guardar resultados
df_resultados.to_csv('diagnosticos_psiquiatricos_procesados.csv', index=False)
if not df_errores.empty:
    df_errores.to_csv('errores_procesamiento.csv', index=False)

# Estadísticas detalladas
print(f"\n✅ PROCESAMIENTO COMPLETADO")
print(f"=" * 50)
print(f"📊 Total de registros procesados: {len(df_resultados)}")
print(f"👥 Pacientes únicos: {df_resultados['Numero_Historia'].nunique()}")
print(f"🏥 Diagnósticos únicos: {df_resultados['Diagnostico'].nunique()}")
print(f"⚠️  Errores encontrados: {len(df_errores)}")

print(f"\n🔍 DIAGNÓSTICOS ÚNICOS ENCONTRADOS:")
print("-" * 40)
diagnosticos_unicos = df_resultados['Diagnostico'].value_counts().sort_index()
for diag, count in diagnosticos_unicos.items():
    porcentaje = (count / len(df_resultados)) * 100
    print(f"{diag}: {count:4d} registros ({porcentaje:5.1f}%)")

print(f"\n📈 TOP 10 DIAGNÓSTICOS MÁS FRECUENTES:")
print("-" * 45)
top_diagnosticos = df_resultados['Diagnostico'].value_counts().head(10)
for diag, count in top_diagnosticos.items():
    porcentaje = (count / len(df_resultados)) * 100
    print(f"{diag}: {count:4d} registros ({porcentaje:5.1f}%)")

# Análisis de pacientes con múltiples diagnósticos
pacientes_multiples = df_resultados.groupby('Numero_Historia').size()
pacientes_con_multiples = pacientes_multiples[pacientes_multiples > 1]

print(f"\n👥 ANÁLISIS DE PACIENTES CON MÚLTIPLES DIAGNÓSTICOS:")
print("-" * 55)
print(f"Pacientes con un solo diagnóstico: {(pacientes_multiples == 1).sum()}")
print(f"Pacientes con múltiples diagnósticos: {len(pacientes_con_multiples)}")
if len(pacientes_con_multiples) > 0:
    print(f"Máximo diagnósticos por paciente: {pacientes_multiples.max()}")
    print(f"Promedio diagnósticos por paciente: {pacientes_multiples.mean():.1f}")

print(f"\n💾 Archivos guardados:")
print(f"• diagnosticos_psiquiatricos_procesados.csv ({len(df_resultados)} registros)")
if not df_errores.empty:
    print(f"• errores_procesamiento.csv ({len(df_errores)} errores)")