package utils

interface IMapper<I,O> {
    fun map(input: I): O
}