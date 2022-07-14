package network

data class NetworkAlbum(
//    @SerializedName("id")
    val id: String?,
//    @SerializedName("title")
    val title: String?,
//    @SerializedName("songs")
    val songs: List<NetworkSong>?
)

data class NetworkSong(
//    @SerializedName("id")
    val id: String?,
//    @SerializedName("name")
    val name: String?,
//    @SerializedName("link")
    val link: String?,
//    @SerializedName("duration")
    val duration: Long?,
//    @SerializedName("creationDate")
    val creationDate: String?,
//    @SerializedName("uploadDate")
    val uploadDate: String?,
//    @SerializedName("authorFullName")
    val authorFullName: String?
)