import React, { Component } from "react";
import styled, { css } from "styled-components";
import EntypoIcon from "react-native-vector-icons/dist/Entypo";

function CameraButton(props) {
  return (
    <Container {...props}>
      <Button>
        <ButtonOverlay >
          <EntypoIcon
            name={props.icon || "camera"}
            style={{
              color: "rgba(255,255,255,1)",
              fontSize: 25,
              height: 27,
              width: 25,
            }}
          ></EntypoIcon>
        </ButtonOverlay>
      </Button>
    </Container>
  );
}

const Container = styled.div`
  display: flex;
  flex-direction: column;
`;

const ButtonOverlay = styled.button`
 display: block;
 background: none;
 height: 100%;
 width: 100%;
 border:none
 `;
const Button = styled.div`
  background-color: rgba(137,0,0,1);
  border-radius: 100px;
  flex-direction: column;
  display: flex;
  flex: 1 1 0%;
  border: none;
`;

export default CameraButton;
