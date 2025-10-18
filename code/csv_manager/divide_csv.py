import pandas as pd
import re

# Leer el archivo CSV
df = pd.read_csv(r"c:\Users\Usuario\Documents\Workspace\Estudio-Psiquiatricos\datasets\neo4j\diagnosticos_por_paciente.csv")

# Lista para almacenar los resultados
resultados = []

def limpiar_y_extraer_diagnosticos(valor):
    """Función para limpiar y extraer diagnósticos de una celda"""
    if pd.isna(valor) or str(valor).strip() in ['', '[]']:
        return []
    
    valor_str = str(valor)
    
    # Remover corchetes externos
    valor_str = re.sub(r'^\[|\]$', '', valor_str)
    
    # Si está vacío después de remover corchetes
    if not valor_str.strip():
        return []
    
    # Dividir por comas y limpiar cada diagnóstico
    diagnosticos = []
    
    # Usar regex para encontrar códigos de diagnóstico
    # Patrón para códigos como F20.0, E11.65, etc.
    patron_diagnosticos = re.findall(r"'([A-Z]\d{1,3}(?:\.\d{1,3})?[A-Z]?)'", valor_str)
    
    # Si no encuentra con el patrón, usar división por comas
    if not patron_diagnosticos:
        # Remover todas las comillas y dividir por comas
        valor_limpio = re.sub(r"['\"]", "", valor_str)
        diagnosticos_temp = [d.strip() for d in valor_limpio.split(',') if d.strip()]
        patron_diagnosticos = [d for d in diagnosticos_temp if re.match(r'^[A-Z]\d', d)]
    
    return patron_diagnosticos

# Procesar cada fila
for index, row in df.iterrows():
    numero_historia = row['Nº Historia']
    
    # Procesar todas las columnas de diagnósticos
    for col in df.columns:
        if 'Diag' in col and 'cod' in col:
            valor = row[col]
            
            # Extraer diagnósticos de la celda
            diagnosticos = limpiar_y_extraer_diagnosticos(valor)
            
            # Agregar cada diagnóstico como una fila separada
            for diagnostico in diagnosticos:
                resultados.append({
                    'Numero_Historia': numero_historia,
                    'Diagnostico': diagnostico
                })

# Crear DataFrame con los resultados
df_resultados = pd.DataFrame(resultados)

# Eliminar duplicados y ordenar
df_resultados = df_resultados.drop_duplicates().sort_values('Numero_Historia').reset_index(drop=True)

# Guardar el resultado
df_resultados.to_csv('diagnosticos_procesados_limpio.csv', index=False)

print(f"✅ Procesamiento completado!")
print(f"📊 Total de registros procesados: {len(df_resultados)}")
print(f"👥 Pacientes únicos: {df_resultados['Numero_Historia'].nunique()}")
print(f"🏥 Diagnósticos únicos: {df_resultados['Diagnostico'].nunique()}")

# Mostrar estadísticas
print(f"\n📋 Primeros registros:")
print(df_resultados.head(10))

print(f"\n📈 Diagnósticos más frecuentes:")
print(df_resultados['Diagnostico'].value_counts().head(15))

print(f"\n🔍 Diagnósticos únicos encontrados:")
diagnosticos_unicos = sorted(df_resultados['Diagnostico'].unique())
for i, diag in enumerate(diagnosticos_unicos[:20]):  # Mostrar solo los primeros 20
    print(f"{i+1:2d}. {diag}")
if len(diagnosticos_unicos) > 20:
    print(f"... y {len(diagnosticos_unicos)-20} más")