class Person {
    constructor(name) {
        this.name = name;
        this.id = Person.nextId++;
    }
}

Person.nextId = 0;

const temp = new Person("temp");

console.log(temp)
