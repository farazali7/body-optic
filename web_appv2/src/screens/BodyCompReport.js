import React, { Component } from "react";
import styled, { css } from "styled-components";
import CameraButton from "../components/CameraButton";
import { Link } from "react-router-dom";

function BodyCompReport(props) {
  return (
    <>
      <BodyFatPercentage placeholder="Body Fat Percentage:"></BodyFatPercentage>
      <BfpValue placeholder="%" editable={false}></BfpValue>
      <ScanAgainStack>
        <ScanAgain>Scan again</ScanAgain>
        <Link to="/MainPage" >
          <CameraButton
            iconName="ccw"
            style={{
              position: "absolute",
              left: 8,
              top: 0,
              width: 54,
              height: 54
            }}
            icon="ccw"
            button="MainPage"
          ></CameraButton>
        </Link>
      </ScanAgainStack>
    </>
  );
}

const BodyFatPercentage = styled.input`
  font-family: Alatsi;
  color: #121212;
  width: 191px;
  height: 40px;
  font-style: normal;
  font-weight: 400;
  font-size: 20px;
  margin-top: 236px;
  margin-left: 92px;
  border: none;
  background: transparent;
`;

const BfpValue = styled.input`
  font-family: Alata;
  font-style: normal;
  font-weight: 400;
  color: rgba(137,0,0,1);
  width: 46px;
  height: 76px;
  font-size: 50px;
  background-color: rgba(255,255,255,1);
  margin-top: 32px;
  margin-left: 165px;
  border: none;
  background: transparent;
`;

const ScanAgain = styled.span`
  font-family: Alata;
  top: 26px;
  left: 0px;
  position: absolute;
  color: rgba(255,255,255,1);
  width: 72px;
  height: 20px;
`;

const ScanAgainStack = styled.div`
  width: 72px;
  height: 54px;
  margin-top: 200px;
  margin-left: 152px;
  position: relative;
`;

export default BodyCompReport;
