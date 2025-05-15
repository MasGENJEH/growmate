export default class DiseaseClassificationPage {
  #takenDocumentation = null;

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
            <div class="bottom-content main-recommendation-content">
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
              <div class="item">
                <form id="form-upload">
                  <div class="upload-button-container">
                    <button class="btn camera-button">Ambil Gambar</button>
                    <label for="image-file" class="btn file-button">Unggah Gambar</label>
                    <input type="file" id="image-file" name="image-file" style="display: none">
                  </div>
                  <div class="image-preview" id="image-preview"></div>
                  <div class="detection-button-container" id="detection-button-container"></div>
                </form>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  async afterRender() {
    this.#setupForm();
  }

  #setupForm() {
    const imageFile = document.querySelector('#image-file');
    const imagePreview = document.querySelector('#image-preview');
    const detectionButtonContainer = document.querySelector('#detection-button-container');

    imageFile.addEventListener('change', () => {
      const file = imageFile.files[0];

      if (!file) return;

      this.#takenDocumentation = {
        blob: file,
        url: URL.createObjectURL(file),
      };

      const img = document.createElement('img');
      img.src = this.#takenDocumentation.url;
      img.alt = 'preview';
      imagePreview.innerHTML = '';
      imagePreview.appendChild(img);

      const detectionButton = document.createElement('button');
      detectionButton.classList.add('detection-button');
      detectionButton.type = 'button';
      detectionButton.textContent = 'Deteksi Sekarang';
      detectionButtonContainer.innerHTML = '';
      detectionButtonContainer.appendChild(detectionButton);
      
      this.#setupDecationButton();
    })

  }

  #setupDecationButton() {
    const detectionButton = document.querySelector('.detection-button');
    detectionButton ? detectionButton.addEventListener('click', (event) => {
      event.stopPropagation();
      const mainRecommendationContent = document.querySelector('.main-recommendation-content');
      const item = document.createElement('div');
      item.classList.add('item');
      const title = document.createElement('p');
      title.classList.add('title');
      title.innerHTML = '🔍 Hasil';
      const paragraph = document.createElement('p');
      paragraph.innerHTML = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Ut ultricies posuere ipsum eget fermentum. Maecenas sodales urna eu accumsan sodales. Duis non ipsum mollis, feugiat sem tempor, dapibus magna.';
      item.appendChild(title);
      item.appendChild(paragraph);
      mainRecommendationContent.appendChild(item);  


      const item2 = document.createElement('div');
      item2.classList.add('item');
      const title2 = document.createElement('p');
      title2.classList.add('title');
      title2.innerHTML = '💡 Saran';
      const paragraph2 = document.createElement('p');
      paragraph2.innerHTML = 'Praesent nec consectetur neque, vel efficitur neque. Vestibulum non turpis a nisl ornare imperdiet vitae et nisi. Duis tincidunt lobortis tellus ac viverra. Sed suscipit varius imperdiet. Ut condimentum pretium odio, non blandit metus viverra id. Proin sollicitudin metus nec volutpat commodo.';
      item2.appendChild(title2);
      item2.appendChild(paragraph2);
      mainRecommendationContent.appendChild(item2);
    }) : null;
  }
}