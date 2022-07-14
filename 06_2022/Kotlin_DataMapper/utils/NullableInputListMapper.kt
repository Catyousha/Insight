package utils

interface INullableInputListMapper<I, O>: IMapper<List<I>?, List<O>>

class NullableInputListMapper<I, O>(
    private val mapper: IMapper<I, O>
): INullableInputListMapper<I, O> {
    override fun map(input: List<I>?): List<O> {
        return input?.map { mapper.map(it) }.orEmpty()
    }

}