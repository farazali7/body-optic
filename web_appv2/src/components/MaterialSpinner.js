import React, { Component } from "react";
import styled, { css } from "styled-components";
import CircularProgress from "@material-ui/core/CircularProgress";

function MaterialSpinner(props) {
  return (
    <Container {...props}>
      <CircularProgress
        style={{
          width: 22,
          height: 22
        }}
      ></CircularProgress>
    </Container>
  );
}

const Container = styled.div`
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
`;

export default MaterialSpinner;
