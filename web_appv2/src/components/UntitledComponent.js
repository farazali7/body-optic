import React, { Component } from "react";
import styled, { css } from "styled-components";

function UntitledComponent(props) {
  return <Container {...props}></Container>;
}

const Container = styled.div`
 background-color: #E6E6E6;
`;

export default UntitledComponent;
