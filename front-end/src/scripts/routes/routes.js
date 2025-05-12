import DashboardPage from "../pages/dashboard/dashboard-page.js";
import GrowmatePage from "../pages/growmate/growmate-page.js";

const routes = {
  '/': new DashboardPage(),
  '/growmate': new GrowmatePage(),
}

export default routes;