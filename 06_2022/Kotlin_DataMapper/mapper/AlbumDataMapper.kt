package mapper

import com.sun.org.apache.xalan.internal.lib.ExsltDatetime.formatDate
import model.Album
import model.Song
import network.NetworkAlbum
import network.NetworkSong
import utils.IMapper
import utils.NullableInputListMapper
import java.text.SimpleDateFormat
import java.util.*

class AlbumDataMapper(
    private val songListDataMapper: NullableInputListMapper<NetworkSong, Song>
): IMapper<NetworkAlbum, Album> {
    override fun map(input: NetworkAlbum): Album {
        return Album(
            input.id.orEmpty(),
            input.title.orEmpty(),
            songListDataMapper.map(input.songs)
        )
    }
}

// usage:
// val album = AlbumDataMapper.map(response)

class SongDataMapper : IMapper<NetworkSong, Song> {
    override fun map(input: NetworkSong): Song {
        return Song(
            input.id.orEmpty(),
            input.name.orEmpty(),
            input.link.orEmpty(),
            input.duration ?: 0,
            Song.Metadata(
                formatDate(input.creationDate),
                formatDate(input.uploadDate),
                input.authorFullName.orEmpty()
            )
        )
    }

    private fun formatDate(date: String?): Long {
        return date?.let {
            SimpleDateFormat(DATE_FORMAT_PATTERN, Locale.ROOT).parse(it).time
        } ?: Long.MIN_VALUE
    }

    private companion object {
        const val DATE_FORMAT_PATTERN = "dd MMMM yyyy HH:mm:ss"
    }

}