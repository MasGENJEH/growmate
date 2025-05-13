export default class DiseaseClassificationPage {
  async render() {
    return `
      <div class="main-feature-content">
        <div class="jumbotron">
          <img src="images/disease-classification-1.jpg" alt="">
          <h1 class="title">Deteksi Penyakit <span>Tanaman</span></h1>
        </div>
        <div class="container">
          <div class="description">
            <p>Pengguna dapat mengidentifikasi penyakit yang menyerang tanaman hanya dengan mengunggah foto daun atau bagian tanaman yang tampak terinfeksi. Sistem berbasis AI akan secara otomatis menganalisis gambar dan menampilkan hasil klasifikasi, lengkap dengan nama penyakit, gejala umum, serta saran penanganan awal yang dapat segera dilakukan oleh petani.</p>
          </div>

          <div class="recomendation">
            <div class="top-content">
              <i class="bi bi-search"></i>
              <p>Rekomendasi</p>
            </div>
            <div class="bottom-content">
              <div class="item">
                <p class="title">🌱 Daftar Penyakit yang Bisa Dideteksi</p>
                <p>Tampilkan jenis-jenis penyakit tanaman umum yang bisa dikenali oleh sistem.</p>
              </div>
              <div class="item">
                <p class="title">🤖 Keunggulan Teknologi yang Digunakan</p>
                <p>Didukung oleh model AI yang telah dilatih dengan ribuan gambar penyakit tanaman untuk akurasi deteksi yang tinggi.</p>
              </div>
              <div class="item">
                <p class="title">📸 Tips Foto yang Efektif untuk Deteksi</p>
                <ul>
                  <li>Ambil foto dengan pencahayaan yang baik.</li>
                  <li>Pastikan daun atau bagian tanaman terlihat jelas.</li>
                  <li>Hindari foto yang terlalu jauh atau terlalu dekat.</li>
                  <li>Usahakan untuk tidak ada objek lain yang menghalangi bagian tanaman yang ingin dideteksi.</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  async afterRender() {}
}