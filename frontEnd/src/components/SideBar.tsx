import * as React from "react";
import {Model} from "../types"
import "./SideBar.css"
import { Menu, Icon, Button } from 'antd';
const SubMenu = Menu.SubMenu;
// const MenuItemGroup = Menu.ItemGroup;
export interface Props{
    onImportModel:(json:Model)=>void
}

class SiderBar extends React.Component<Props, any> {
    public file: any;
    constructor(pros: any) {
        super(pros)
        this.onImportModel = this.onImportModel.bind(this)
    }
    handleClick = (e: any) => {
        switch (e.key) {
            case "xxx":
                break;
            default:
                break;
        }
        console.log('click ', e);
    }
    onImportModel() {
        let file: any = this.file.files[0]
        let reader = new FileReader()
        reader.onload = (e: any) => {
            this.props.onImportModel(JSON.parse(e.target.result))
            // console.info(JSON.parse(e.target.result))
        }
        reader.readAsText(file)
    }
    render() {
        return (
            <div className="sideBar">
                <span className="menuItem ant-menu-item ">
                    <div className="fileinputs">
                        <input type="file" 
                        className="file" 
                        ref={(ref) => this.file = ref}
                        onChange={this.onImportModel}/>

                        <div className="fakefile">
                            {/* <input type="button" value="Import Model" /> */}
                            <Button className="inputButton ant-menu-dark">
                                <span className="menuItem"><Icon type="folder-open" />Import Model</span>
                            </Button>
                        </div>
                    </div>
                </span>
                <Menu
                    theme="dark"
                    onClick={this.handleClick}
                    defaultSelectedKeys={[]}
                    defaultOpenKeys={['add_layers', 'conv_layers']}
                    mode="inline"
                >

                    <SubMenu key="add_layers" title={<span className="menuItem"><Icon type="barcode" />Add Layer</span>}>
                        <SubMenu key="core_layers" title={<span className="layerClass">Core Layers</span>}>
                            <Menu.Item key="1"><span className="layerName">Layer 1</span></Menu.Item>
                            <Menu.Item key="2">Layer 2</Menu.Item>
                        </SubMenu>
                        <SubMenu key="conv_layers" title={<span className="layerClass">Conv Layers</span>}>
                            <Menu.Item key="3">Layer 3</Menu.Item>
                            <Menu.Item key="4">Layer 4</Menu.Item>
                        </SubMenu>
                    </SubMenu>
                </Menu>
            </div>
        );
    }
}

export default SiderBar