import React from "react";
import { BrowserRouter as Router, Route } from "react-router-dom";
import "./icons.js";
import MainPage from "./screens/MainPage";
import Calculation from "./screens/Calculation";
import BodyCompReport from "./screens/BodyCompReport";
import "./style.css";

function App() {
  return (
    <Router>
      <Route path="/" exact component={MainPage} />
      <Route path="/MainPage/" exact component={MainPage} />
      <Route path="/Calculation/" exact component={Calculation} />
      <Route path="/BodyCompReport/" exact component={BodyCompReport} />
    </Router>
  );
}

export default App;
