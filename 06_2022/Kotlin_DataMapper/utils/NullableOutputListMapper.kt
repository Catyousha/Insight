package utils

interface INullableOutputListMapper<I, O>: IMapper<List<I>, List<O>?>

class NullableOutputListMapper<I, O>(
    private val mapper: IMapper<I, O>
): INullableOutputListMapper<I, O> {
    override fun map(input: List<I>): List<O>? {
        return if(input.isEmpty()) null else input.map { mapper.map(it) }
    }
}