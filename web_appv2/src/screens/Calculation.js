import React, { Component } from "react";
import styled, { css } from "styled-components";
import MaterialSpinner from "../components/MaterialSpinner";

function Calculation(props) {
  return (
    <Stack>
      <MaterialSpinner
        style={{
          height: 131,
          position: "absolute",
          top: 0,
          width: 169,
          left: 0
        }}
      ></MaterialSpinner>
      <CalculatingTxt placeholder="Calculating..."></CalculatingTxt>
    </Stack>
  );
}

const Stack = styled.div`
  width: 169px;
  height: 131px;
  margin-top: 298px;
  margin-left: 103px;
  position: relative;
  display: flex;
`;

const CalculatingTxt = styled.input`
  font-family: Alatsi;
  top: 6px;
  left: 25px;
  position: absolute;
  font-style: normal;
  font-weight: 400;
  color: #121212;
  width: 118px;
  height: 44px;
  font-size: 20px;
  border: none;
  background: transparent;
`;

export default Calculation;
