import DashboardPage from "../pages/dashboard/dashboard-page.js";
import DiseaseClassificationPage from "../pages/disease-classification/disease-classification-page.js";
import GrowmatePage from "../pages/growmate/growmate-page.js";

const routes = {
  '/': () => new DashboardPage(),
  '/growmate': () => new GrowmatePage(),
  '/disease-classification': () => new DiseaseClassificationPage()
}

export default routes;