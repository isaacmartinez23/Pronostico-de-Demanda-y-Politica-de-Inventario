# Resumen ejecutivo — Pronóstico de demanda y política de inventario

**Pregunta:** ¿cuánto vamos a vender de cada producto las próximas 4 semanas y cuánto
inventario necesitamos para atender el 95% de la demanda sin sobre-stock?

**Alcance:** 1,429 productos de alimentos de una tienda Walmart (dataset público M5), con
cinco años de ventas diarias, precios y calendario de eventos.

## Resultado

> Para atender el 95% de la demanda con 7 días de lead time, la política basada en el
> modelo necesita **14% menos inventario promedio** que una política clásica de media
> móvil: USD 63.7k contra USD 73.8k, a costo.

![Curva servicio–inventario](figures/16_frontera_servicio_inventario.png)

| Indicador | Resultado |
| --- | --- |
| Error de pronóstico (WRMSSE) | 33% menor que repetir la semana anterior; 13% menor que el mejor método estadístico |
| Fill rate en productos clase A (80% del ingreso) | 96.8%, sobre el objetivo de 95% |
| Inventario a igual servicio (95% de fill rate) | −14% en total; −11% en clase A; −25% en clase B |
| Sensibilidad al lead time | −7% con 3 días, −14% con 7, −22% con 14 |

La comparación se hizo simulando ambas políticas día a día contra la demanda real de 12
semanas que el modelo no vio, y midiendo cuánto inventario necesita cada una para llegar
al mismo nivel de servicio.

## Qué significa para la operación

1. **El beneficio crece cuanto más lejos hay que planear.** Con proveedores de 14 días el
   ahorro supera el 20%; con reposición de 3 días es de un dígito.
2. **No todos los productos necesitan un modelo.** El modelo gana en los productos de mayor
   venta (segmentos AX y AY, tres cuartas partes del ingreso). En la cola de baja rotación
   (clase C) un promedio simple pronostica igual o mejor.
3. **El "95%" de la fórmula no es el 95% que ve el cliente.** Con el factor de seguridad de
   libro para 95%, el servicio por ciclo de reposición real fue de 90%. El factor de
   seguridad debe fijarse por simulación contra el fill rate deseado.
4. **Refinar la fórmula del safety stock no ahorró inventario.** Dos alternativas más
   sofisticadas dieron la misma eficiencia: lo que mueve el resultado es la calidad del
   pronóstico.

## Recomendación

Adoptar el pronóstico del modelo para las clases A y B, mantener una regla de media móvil
para la clase C, y calibrar el factor de seguridad por clase con la simulación. Empezar por
los productos con lead time largo, donde está la mayor parte del ahorro.

## Límites del análisis

- Los datos registran ventas, no demanda: los quiebres históricos no se ven.
- Los costos (unitario, de mantener y por pedido) son supuestos; al variarlos, el ahorro se
  mantuvo entre 13% y 14%.
- Lead time fijo, sin mínimos de compra, capacidad ni caducidad.
- Es una tienda y una categoría; el resultado debe confirmarse en otras antes de generalizar.
