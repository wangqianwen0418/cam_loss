export interface StoreState {
    model:Model;
    ids:number[];
    bbox:boolean;
    imgset:"val"|"train";
    [key:string]:any;
}

export interface Model {
    node:Array<any>
}