export default class GrowmatePage {
  async render() {
    return `<div class="growmate-content">
      <div class="container">
        <div class="jumbotron">
          <div class="description">
            <h1>GrowMate</h1>
            <p>GrowMate adalah aplikasi pintar yang membantu petani dan pecinta tanaman dalam mengidentifikasi hama, penyakit, serta memberikan rekomendasi perawatan secara akurat dan cepat.</p>
          </div>
          <div class="image-container">
            <img src="images/person.png" alt="">
            <div class="bubble"></div>
            <div class="bubble"></div>
            <div class="bubble"></div>
          </div>
        </div>

        <div class="main-content"></div>
      </div>
    </div>`
  }

  async afterRender() {
    // Add any additional functionality or event listeners here
    
  }
}