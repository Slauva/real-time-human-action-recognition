

export class MLModel {
    id: string;
    name: string;

    constructor(data: any) {
        this.id = data.id;
        this.name = data.name;
    }

    toString(): string {
        return this.name;
    }
}