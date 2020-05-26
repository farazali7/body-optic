import React, { Component } from "react";
import styled, { css } from "styled-components";
import CameraButton from "../components/CameraButton";
import IconButton from "@material-ui/core/IconButton";
import PhotoCamera from "@material-ui/icons/PhotoCamera";
import ImageIcon from "@material-ui/icons/Image";
import axios from "axios";

export class MainPage extends Component {
  state ={
    selectedImage: null
  }

  imageSelectedHandler = event => {
    console.log(event.target.files[0])
    this.setState({
      selectedImage: URL.createObjectURL(event.target.files[0])
    })

    const formData = new FormData();
    formData.append("image", this.state.selectedImage)

    axios.post("http://localhost:5000/send_image", formData)
  }
  
  render() {
    return (
      <>
        <BodyOpticLogo
          src={require("../assets/images/logo1.png")}
        ></BodyOpticLogo>
        <div>
          <input
              accept="image/*"
              id="upload-image"
              type="file"
              style={{display:'none'}}
              onChange={this.imageSelectedHandler}
            />
          <label htmlFor="upload-image" 
            style={{
              height: 54,
              width: 54,
              marginTop: 45,
              marginLeft: 160
            }}>
            <IconButton component='span'>
              <PhotoCamera/>
            </IconButton>    
          </label>
        </div>
        <div>
          <input
              accept="image/*"
              id="upload-image"
              type="file"
              style={{display:'none'}}
            />
          <label htmlFor="upload-image" 
            style={{
              height: 54,
              width: 54,
              marginTop: 45,
              marginLeft: 160
            }}>
            <IconButton component='span'>
              <ImageIcon/>
            </IconButton>    
          </label>
        </div>
      </>
    );
  }
}

const BodyOpticLogo = styled.img`
  width: 200px;
  height: 200px;
  margin-top: 112px;
  margin-left: 87px;
  object-fit: contain;
`;

export default MainPage;
